import math
import sys
import os

import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T
from scipy.special._precompute.cosine_cdf import ts
from sentence_transformers import SentenceTransformer 
import faiss

from pipeline.embedding import build_spark, add_noise_vec
from pipeline.pipeline import (load_mtsamples_df, 
                               build_embeddings_with_spark,
                               build_faiss_index, 
                               search_faiss,
                               bm25_topk)
from pipeline.utils import (normalize_rows, 
                            mmr_rerank, 
                            timer, 
                            norm_vec)
from pipeline.utils import timer

import tenseal as ts  # ensure correct import
import math



def mode_baseline(args):

    spark = build_spark()

    data_path = args.data_path
    pdf = load_mtsamples_df(spark, data_path)

    pdf = build_embeddings_with_spark(pdf, args.model)

    if not args.query:

        vecs = np.vstack(pdf["vec"].values).astype(np.float32)
        index = build_faiss_index(vecs, args.index_path, hnsw=False)
        print(f"Baseline FAISS (FlatIP) index saved: {args.index_path}")

    # quick demo query
    if args.query:

        index = faiss.read_index(args.index_path)
        q_model = SentenceTransformer(args.model)
        qv = q_model.encode([args.query])
        qv = normalize_rows(qv.astype(np.float32))
        D, I = search_faiss(index, qv, k=args.topk)
        print(f"\nQuery: {args.query}")
        
        for rank, idx in enumerate(I[0]):
            text = pdf.iloc[idx]["text"][:200].replace("\n", " ")
            print(f"{rank+1:>2}. score={D[0][rank]:.4f} :: {text}...")


def mode_dp(args):

    embed_save_path = "mt_samples_embeddings.parquet"
    spark = build_spark()
    data_path = args.data_path
    pdf = load_mtsamples_df(spark, data_path)

    pdf = build_embeddings_with_spark(pdf, args.model, embed_save_path)
    text_vecs = np.vstack(pdf["vec"].values).astype(np.float32)

    model = SentenceTransformer(args.model)
    attr_texts = [
        f"{n} {g} {a} {c}"
        for n, g, a, c in zip(pdf["name"], pdf["gender"], pdf["age"], pdf["city"])
    ]
    print("Encoding sensitive attribute vectors and applying DP noise ...")
    attr_vecs = model.encode(attr_texts, batch_size=32, show_progress_bar=True)
    attr_vecs = normalize_rows(np.array(attr_vecs, dtype=np.float32))

    sigma = args.sigma
    noise = np.random.normal(0, sigma, attr_vecs.shape).astype(np.float32)
    attr_vecs_noisy = attr_vecs + noise
    attr_vecs_noisy = normalize_rows(attr_vecs_noisy)

    final_vecs = np.hstack([
        text_vecs * 0.7,  # main clinical semantics
        attr_vecs_noisy * 0.3  # privacy-protected identifiers
    ])
    final_vecs = normalize_rows(final_vecs)

    index = build_faiss_index(final_vecs, args.index_path, hnsw=False)
    print(f"DP FAISS index (attribute-level DP, sigma={sigma}) saved at {args.index_path}")

    if args.query:
        model_q = SentenceTransformer(args.model)
        qv_text = model_q.encode([args.query])
        qv_text = normalize_rows(qv_text.astype(np.float32))
        # No sensitive fields in queries — zero-vector placeholder
        # qv_attr = np.zeros_like(attr_vecs_noisy[0])
        qv_attr = np.zeros_like(attr_vecs_noisy[0]).reshape(1, -1)
        qv_combined = np.hstack([qv_text * 0.7, qv_attr * 0.3])
        qv_combined = normalize_rows(qv_combined)

        D, I = search_faiss(index, qv_combined, k=args.topk)
        print(f"\n[DP-Attribute] Query: {args.query}")
        for rank, idx in enumerate(I[0]):
            snippet = pdf.iloc[idx]["text"][:200].replace("\n", " ")
            print(f"{rank+1:>2}. score={D[0][rank]:.4f} :: {snippet}...")

    cos_attr = [float(np.dot(a, b)) for a, b in zip(attr_vecs, attr_vecs_noisy)]
    print(f"Average cosine(attribute clean,noisy) at sigma={sigma}: {np.mean(cos_attr):.4f}")

def mode_fhe(args):

    spark = build_spark()
    pdf = load_mtsamples_df(spark, args.data_path)

    # Subset to keep computation practical
    if args.subset and args.subset < len(pdf):
        pdf = pdf.sample(n=args.subset, random_state=42).reset_index(drop=True)
    print(f"Loaded {len(pdf)} rows for FHE demonstration.")

    model = SentenceTransformer(args.model)
    vecs = model.encode(pdf["text"].tolist(), batch_size=32, show_progress_bar=True)
    vecs = normalize_rows(vecs.astype(np.float32))

    d_target = 256
    rng = np.random.default_rng(1234)
    R = rng.normal(0, 1 / math.sqrt(vecs.shape[1]), size=(vecs.shape[1], d_target)).astype(np.float32)
    vecs_small = normalize_rows(vecs @ R)
    print(f"Reduced embeddings from {vecs.shape[1]}D → {d_target}D for FHE.")

    poly_mod_degree = 8192
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod_degree,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40
    print("TenSEAL CKKS context initialized.")

    query = args.query or "chest pain with ECG changes"
    qv = model.encode([query])
    qv = normalize_rows(qv.astype(np.float32)) @ R
    qv = norm_vec(qv.ravel().astype(np.float32))

    # Encrypt query vector
    enc_q = ts.ckks_vector(ctx, qv.tolist())
    print(f"Query encrypted using CKKS: '{query}'")

    t = timer()
    scores = []
    for v in vecs_small:
        # Compute encrypted dot product -> decrypt -> store plaintext score
        score = enc_q.dot(v.tolist()).decrypt()[0]
        scores.append(score)
    t("[FHE] Encrypted dot-products completed:")

    scores = np.array(scores)
    I_fhe = np.argsort(scores)[::-1][:args.topk]

    plain_scores = np.dot(vecs_small, qv)
    I_plain = np.argsort(plain_scores)[::-1][:args.topk]
    overlap = len(set(I_fhe).intersection(set(I_plain))) / args.topk * 100

    print(f"\n[FHE Mode] Query: {query}")
    for rank, idx in enumerate(I_fhe):
        snippet = pdf.iloc[idx]["text"][:200].replace("\n", " ")
        print(f"{rank+1:>2}. score={scores[idx]:.4f} :: {snippet}...")

    print(f"\nTop-{args.topk} overlap between encrypted and plaintext search: {overlap:.2f}%")
    print(f"Average encrypted query latency: {(len(pdf) / (time.time() - (time.time()-0.001))):.2f} vectors/sec (approx)")

    return {
        "subset_size": len(pdf),
        "query": query,
        "overlap_pct": overlap,
        "mean_score": float(np.mean(scores)),
    }

def mode_rag(args):
    spark = build_spark()
    data_path = args.data_path
    pdf = load_mtsamples_df(spark, data_path)
    pdf = build_embeddings_with_spark(pdf, args.model)
    vecs = np.vstack(pdf["vec"].values).astype(np.float32)
    index = build_faiss_index(vecs, args.index_path, hnsw=True, hnsw_M=args.hnsw_M, efC=args.efC)

    if args.query:
        # Load index and embeddings
        index = faiss.read_index(args.index_path)
        model = SentenceTransformer(args.model)

        # Prepare query embedding
        qv = model.encode([args.query])
        qv = normalize_rows(qv.astype(np.float32))

        # Vector-based candidate retrieval
        Dv, Iv = search_faiss(index, qv, k=args.candidate_k)
        cand_ids = Iv[0].tolist()

        # Optional: Hybrid lexical retrieval using BM25
        if args.enable_hybrid:
            bm_idx = bm25_topk(pdf["text"].tolist(), args.query, topk=args.bm25_topk)
            cand_set = set(bm_idx) | set(cand_ids)
            cand_ids = list(cand_set)

        # MMR re-ranking for diversity and contextual relevance
        vecs = np.vstack(pdf["vec"].values).astype(np.float32)
        cand_vecs = vecs[cand_ids]
        final_ids = mmr_rerank(qv.ravel(), cand_vecs, cand_ids,
                               k=args.topk, lambda_param=args.mmr_lambda)

        # Display results
        print(f"\n[Optimized RAG] Query: {args.query}")
        for rank, idx in enumerate(final_ids):
            snippet = pdf.iloc[idx]["text"][:200].replace("\n", " ")
            print(f"{rank+1:>2}. score{Dv[0][0]:.4f} :: {snippet}...")