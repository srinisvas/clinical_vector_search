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
                               search_faiss)
                            #    bm25_topk)
from pipeline.utils import (normalize_rows, 
                            mmr_rerank, 
                            timer, 
                            norm_vec)


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

    spark = build_spark()
    data_path = args.data_path
    pdf = load_mtsamples_df(spark, data_path)

    pdf = build_embeddings_with_spark(pdf, args.model)
    text_vecs = np.vstack(pdf["vec"].values).astype(np.float32)

    model = SentenceTransformer(args.model)
    attr_texts = [
        f"{n} {g} {a} {c}"
        for n, g, a, c in zip(pdf["Name"], pdf["Gender"], pdf["Age"], pdf["City"])
    ]
    print("Encoding sensitive attribute vectors and applying DP noise ...")
    attr_vecs = model.encode(attr_texts, batch_size=32, show_progress_bar=True)
    attr_vecs = normalize_rows(np.array(attr_vecs, dtype=np.float32))

    sigma = args.sigma
    noise = np.random.normal(0, sigma, attr_vecs.shape).astype(np.float32)
    attr_vecs_noisy = attr_vecs + noise
    attr_vecs_noisy = normalize_rows(attr_vecs_noisy)

    final_vecs = np.hstack([
        text_vecs * 0.9,  # main clinical semantics
        attr_vecs_noisy * 0.1  # privacy-protected identifiers
    ])
    final_vecs = normalize_rows(final_vecs)

    index = build_faiss_index(final_vecs, args.index_path, hnsw=False)
    print(f"DP FAISS index (attribute-level DP, σ={sigma}) saved at {args.index_path}")

    if args.query:
        model_q = SentenceTransformer(args.model)
        qv_text = model_q.encode([args.query])
        qv_text = normalize_rows(qv_text.astype(np.float32))
        # No sensitive fields in queries — zero-vector placeholder
        qv_attr = np.zeros_like(attr_vecs_noisy[0])
        qv_combined = np.hstack([qv_text * 0.9, qv_attr * 0.1])
        qv_combined = normalize_rows(qv_combined)

        D, I = search_faiss(index, qv_combined, k=args.topk)
        print(f"\n[DP-Attribute] Query: {args.query}")
        for rank, idx in enumerate(I[0]):
            snippet = pdf.iloc[idx]["text"][:200].replace("\n", " ")
            print(f"{rank+1:>2}. score={D[0][rank]:.4f} :: {snippet}...")

    cos_attr = [float(np.dot(a, b)) for a, b in zip(attr_vecs, attr_vecs_noisy)]
    print(f"Average cosine(attribute clean,noisy) @ σ={sigma}: {np.mean(cos_attr):.4f}")

def mode_fhe(args):
    # if not HAS_TENSEAL:
    #     print("TenSEAL not installed. `pip install tenseal` to run FHE POC.")
    #     sys.exit(1)

    spark = build_spark()
    pdf = load_mtsamples_df(spark, args.data_path)
    if args.subset and args.subset < len(pdf):
        pdf = pdf.sample(n=args.subset, random_state=42).reset_index(drop=True)
    print(f"Loaded rows for FHE POC: {len(pdf)}")

    # Smaller model (or PCA) recommended for speed; we’ll project to 256-d
    model = SentenceTransformer(args.model)
    vecs = model.encode(pdf["text"].tolist(), batch_size=32, show_progress_bar=True)
    vecs = normalize_rows(vecs.astype(np.float32))

    # Project to 256 dims with random projection (fast + simple)
    d_target = 256
    rng = np.random.default_rng(1234)
    R = rng.normal(0, 1 / math.sqrt(vecs.shape[1]), size=(vecs.shape[1], d_target)).astype(np.float32)
    vecs_small = normalize_rows(vecs @ R)

    # Prepare TenSEAL CKKS context
    poly_mod = 8192
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_mod,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.generate_galois_keys()
    ctx.global_scale = 2 ** 40

    # Pick a query
    query = args.query or "chest pain with ECG changes"
    qv = model.encode([query])
    qv = normalize_rows(qv.astype(np.float32)) @ R
    qv = norm_vec(qv.ravel().astype(np.float32))

    # Encrypt query only (query-only encryption)
    enc_q = ts.ckks_vector(ctx, qv.tolist())

    # Compute encrypted dot-products with plaintext db vectors
    # NOTE: For a larger set, you’d batch or switch to server-side loops.
    t = timer()
    scores = [enc_q.dot(row.tolist()).decrypt()[0] for row in vecs_small]
    t("[FHE] Encrypted dot-products on subset:")

    # Rank & show
    I = np.argsort(scores)[::-1][:args.topk]
    print(f"\n[FHE POC] Query: {query}")
    for rank, idx in enumerate(I):
        text = pdf.iloc[idx]["text"][:200].replace("\n", " ")
        print(f"{rank+1:>2}. score={scores[idx]:.4f} :: {text}...")


def mode_rag(args):
    spark = build_spark()
    pdf = load_mtsamples_df(spark, args.data_path)
    print(f"Loaded rows: {len(pdf)}")

    # Embeddings
    model = SentenceTransformer(args.model)
    vecs = model.encode(pdf["text"].tolist(), batch_size=32, show_progress_bar=True)
    vecs = normalize_rows(vecs.astype(np.float32))
    pdf["vec"] = list(vecs)

    # Index: HNSW for speed
    index = build_faiss_index(vecs, args.index_path, hnsw=True, hnsw_M=args.hnsw_M, efC=args.efC)
    # At search time, you can set efSearch:
    try:
        index.hnsw.efSearch = args.efS
    except Exception:
        pass
    print(f"HNSW index saved: {args.index_path}")

    # Query
    if not args.query:
        args.query = "post-operative knee arthroscopy pain management"

    qv = model.encode([args.query])
    qv = normalize_rows(qv.astype(np.float32))
    Dv, Iv = search_faiss(index, qv, k=args.candidate_k)  # vector candidates

    # Hybrid (optional BM25)
    if args.enable_hybrid:
        bm_idx = bm25_topk(pdf["text"].tolist(), args.query, topk=args.bm25_topk)
        # Union of candidates
        cand_set = set(bm_idx) | set(Iv[0].tolist())
        cand_ids = list(cand_set)
    else:
        cand_ids = Iv[0].tolist()

    cand_vecs = vecs[cand_ids]
    # MMR re-rank
    final_ids = mmr_rerank(qv.ravel(), cand_vecs, cand_ids, k=args.topk, lambda_param=args.mmr_lambda)

    print(f"\n[RAG] Query: {args.query}")
    for rank, idx in enumerate(final_ids):
        text = pdf.iloc[idx]["text"][:220].replace("\n", " ")
        print(f"{rank+1:>2}. :: {text}...")