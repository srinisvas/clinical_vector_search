import math
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import tenseal as ts

from pipeline.embedding import build_spark
from pipeline.pipeline import (
    load_mtsamples_df,
    build_embeddings_with_spark,
    build_faiss_index,
    search_faiss,
    bm25_topk,
)
from pipeline.utils import normalize_rows, mmr_rerank, timer, norm_vec


# ============================================================
# BASELINE MODE
# ============================================================

def mode_baseline(args):
    """
    Baseline vector search using FlatIP (cosine) FAISS.
    """
    spark = build_spark()

    pdf = load_mtsamples_df(spark, args.data_path)
    pdf = build_embeddings_with_spark(pdf, args.model)

    if not args.query:
        vecs = np.vstack(pdf["vec"].values).astype(np.float32)
        index = build_faiss_index(vecs, args.index_path, hnsw=False)
        print(f"Baseline FAISS index saved at: {args.index_path}")
        return

    # Query mode
    index = faiss.read_index(args.index_path)
    q_model = SentenceTransformer(args.model)
    qv = q_model.encode([args.query])
    qv = normalize_rows(qv.astype(np.float32))

    D, I = search_faiss(index, qv, k=args.topk)

    print(f"\n[BASELINE] Query: {args.query}")
    for rank, idx in enumerate(I[0]):
        snippet = pdf.iloc[idx]["text"][:200].replace("\n", " ")
        print(f"{rank+1:>2}. score={D[0][rank]:.4f} :: {snippet}...")


# ============================================================
# DIFFERENTIAL PRIVACY MODE
# ============================================================

def mode_dp(args):
    """
    DP-enhanced retrieval with noisy sensitive attribute vectors.
    """
    spark = build_spark()

    pdf = load_mtsamples_df(spark, args.data_path)
    pdf = build_embeddings_with_spark(pdf, args.model, "mt_samples_embeddings.parquet")

    text_vecs = np.vstack(pdf["vec"].values).astype(np.float32)

    # Build attribute embeddings
    model = SentenceTransformer(args.model)
    attr_texts = [
        f"{n} {g} {a} {c}"
        for n, g, a, c in zip(pdf["name"], pdf["gender"], pdf["age"], pdf["city"])
    ]
    print("Encoding sensitive attribute vectors...")
    attr_vecs = model.encode(attr_texts, batch_size=32, show_progress_bar=True)
    attr_vecs = normalize_rows(np.array(attr_vecs, dtype=np.float32))

    # Add Gaussian noise
    sigma = args.sigma
    noise = np.random.normal(0, sigma, attr_vecs.shape).astype(np.float32)
    attr_vecs_noisy = normalize_rows(attr_vecs + noise)

    # Combine main text embedding + noisy attributes
    final_vecs = normalize_rows(np.hstack([
        text_vecs * 0.7,
        attr_vecs_noisy * 0.3
    ]))

    index = build_faiss_index(final_vecs, args.index_path, hnsw=False)
    print(f"DP FAISS index saved at: {args.index_path}")

    # Query mode
    if args.query:
        q_model = SentenceTransformer(args.model)
        qv_text = q_model.encode([args.query])
        qv_text = normalize_rows(qv_text.astype(np.float32))

        # Zero attribute for queries
        attr_dim = final_vecs.shape[1] - text_vecs.shape[1]
        qv_attr = np.zeros((1, attr_dim), dtype=np.float32)

        qv = np.hstack([qv_text * 0.7, qv_attr * 0.3])
        qv = normalize_rows(qv)

        D, I = search_faiss(index, qv, k=args.topk)

        print(f"\n[DP Mode] Query: {args.query}")
        for rank, idx in enumerate(I[0]):
            snippet = pdf.iloc[idx]["text"][:200].replace("\n", " ")
            print(f"{rank+1:>2}. score={D[0][rank]:.4f} :: {snippet}...")

    # Noise quality metric
    cos_attr = [float(np.dot(a, b)) for a, b in zip(attr_vecs, attr_vecs_noisy)]
    print(f"Average cosine(original noisy): {np.mean(cos_attr):.4f}")


# ============================================================
# FHE MODE (TenSEAL)
# ============================================================

def mode_fhe(args):
    """
    Fully Homomorphic Encryption search demo.
    Uses TenSEAL CKKS to encrypt the query and compute encrypted dot products.
    """
    spark = build_spark()
    pdf = load_mtsamples_df(spark, args.data_path)

    # Reduce dataset for FHE feasibility
    if args.subset and args.subset < len(pdf):
        pdf = pdf.sample(n=args.subset, random_state=42).reset_ind