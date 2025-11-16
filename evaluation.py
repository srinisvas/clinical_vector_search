#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation.py  —  macOS-compatible evaluation for privacy-preserving retrieval

Compares 4 retrieval modes:
  1. Baseline (FAISS FlatIP)
  2. Differential Privacy (DP)
  3. Fully Homomorphic Encryption (FHE)
  4. Optimized / RAG (HNSW hybrid)

Computes metrics:
  - Latency (ms)
  - Recall@10
  - Precision@10
  - nDCG@10
  - MRR
  - Overlap%
and outputs both CSV and plots under ./plots/

Author: Srinivasan Subramanian (adapted for macOS)
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss
import tenseal as ts
from sklearn.metrics import ndcg_score

from pipeline.utils import normalize_rows, norm_vec

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
QUERIES = [
    "post-operative knee arthroscopy pain management",
    "chest pain with ECG changes",
    "abdominal pain after appendectomy",
    "shortness of breath with asthma history",
    "lumbar spine MRI findings",
    "arthroscopic shoulder repair recovery",
    "ECG abnormalities in heart attack",
    "knee joint effusion",
    "diabetic foot ulcer treatment",
    "post-surgical infection management",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def run_timed(func, *args, **kwargs):
    """Time any function and return (result, elapsed_ms)."""
    start = time.time()
    result = func(*args, **kwargs)
    return result, (time.time() - start) * 1000.0


def recall_at_k(base, other):
    return len(set(base).intersection(set(other))) / max(1, len(base))


def precision_at_k(base, other):
    return len(set(base).intersection(set(other))) / max(1, len(other))


def accuracy_at_k(base, other):
    matches = sum(1 for a, b in zip(base, other) if a == b)
    return matches / max(1, len(base))


def mrr(base, other):
    for rank, doc_id in enumerate(other, start=1):
        if doc_id in base:
            return 1.0 / rank
    return 0.0


def overlap_ratio(base, other):
    """Fractional overlap in retrieved document IDs."""
    return len(set(base).intersection(set(other))) / max(1, len(set(base).union(set(other))))


# ------------------------------------------------------------
# Evaluation Core
# ------------------------------------------------------------
def benchmark_all(data_path, model_name, index_paths):
    """Evaluate all retrieval modes on macOS (single-threaded FAISS + TenSEAL)."""
    print("\nInitializing environment...")
    faiss.omp_set_num_threads(1)

    model = SentenceTransformer(model_name)

    # ---- Load FAISS indexes ----
    print("\nLoading FAISS indexes...")
    baseline_index = faiss.read_index(index_paths["baseline"])
    dp_index = faiss.read_index(index_paths["dp"])
    rag_index = faiss.read_index(index_paths["rag"])

    # ---- Load vectors for FHE projection ----
    print("\nExtracting vectors from baseline index (for FHE projection)...")
    xb = faiss.vector_to_array(baseline_index.reconstruct_n(0, baseline_index.ntotal)).reshape(
        baseline_index.ntotal, baseline_index.d
    )
    xb = normalize_rows(xb)
    fhe_subset = min(200, xb.shape[0])

    results = []

    for query in QUERIES:
        print("\n-------------------------------------------")
        print(f"Evaluating query: {query}")

        # Encode query
        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # 1️⃣ Baseline
        (_, lat_base) = run_timed(baseline_index.search, qv, 10)
        _, I_base = baseline_index.search(qv, 10)
        base_ids = I_base[0]

        # 2️⃣ DP Mode
        dp_dim = dp_index.d
        text_dim = qv.shape[1]
        if dp_dim > text_dim:
            attr_dim = dp_dim - text_dim
            qv_attr = np.zeros((1, attr_dim), dtype=np.float32)
            qv_dp = normalize_rows(np.hstack([qv * 0.7, qv_attr * 0.3]))
        else:
            qv_dp = qv
        (_, lat_dp) = run_timed(dp_index.search, qv_dp, 10)
        _, I_dp = dp_index.search(qv_dp, 10)
        dp_ids = I_dp[0]

        # 3️⃣ FHE Mode (real reduced embeddings)
        d_target = 256
        R = np.random.normal(0, 1 / np.sqrt(qv.shape[1]), size=(qv.shape[1], d_target)).astype(np.float32)
        qv_small = normalize_rows(qv @ R)
        qv_small = norm_vec(qv_small.ravel().astype(np.float32))

        # Project embeddings and normalize
        xb_small = normalize_rows(xb[:fhe_subset] @ R)

        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40
        enc_q = ts.ckks_vector(ctx, qv_small.tolist())

        fhe_scores = []
        t0 = time.time()
        for v in xb_small:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0
        fhe_ids = np.argsort(fhe_scores)[::-1][:10]

        # 4️⃣ RAG (HNSW / hybrid)
        (_, lat_rag) = run_timed(rag_index.search, qv, 10)
        _, I_rag = rag_index.search(qv, 10)
        rag_ids = I_rag[0]

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------
        y_true = np.zeros((1, len(xb)))
        y_score_base = np.zeros((1, len(xb)))
        y_score_rag = np.zeros((1, len(xb)))

        y_true[0, base_ids] = 1
        y_score_base[0, base_ids] = np.linspace(1, 0.1, len(base_ids))
        y_score_rag[0, rag_ids[: len(base_ids)]] = np.linspace(1, 0.1, len(base_ids))
        ndcg_rag = ndcg_score(y_true, y_score_rag)

        results.append(
            {
                "query": query,
                "baseline_latency_ms": lat_base,
                "dp_latency_ms": lat_dp,
                "fhe_latency_ms": lat_fhe,
                "rag_latency_ms": lat_rag,
                "dp_recall": recall_at_k(base_ids, dp_ids),
                "fhe_recall": recall_at_k(base_ids, fhe_ids),
                "rag_recall": recall_at_k(base_ids, rag_ids),
                "dp_precision": precision_at_k(base_ids, dp_ids),
                "fhe_precision": precision_at_k(base_ids, fhe_ids),
                "rag_precision": precision_at_k(base_ids, rag_ids),
                "dp_accuracy": accuracy_at_k(base_ids, dp_ids),
                "fhe_accuracy": accuracy_at_k(base_ids, fhe_ids),
                "rag_accuracy": accuracy_at_k(base_ids, rag_ids),
                "dp_mrr": mrr(base_ids, dp_ids),
                "fhe_mrr": mrr(base_ids, fhe_ids),
                "rag_mrr": mrr(base_ids, rag_ids),
                "dp_overlap": overlap_ratio(base_ids, dp_ids),
                "fhe_overlap": overlap_ratio(base_ids, fhe_ids),
                "rag_overlap": overlap_ratio(base_ids, rag_ids),
                "rag_ndcg": ndcg_rag,
            }
        )

    df = pd.DataFrame(results)
    out_csv = os.path.join(PROJECT_ROOT, "evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved evaluation results to {out_csv}")
    return df


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def generate_plots(df):
    avg = df.mean(numeric_only=True)

    # ---- Latency ----
    plt.figure(figsize=(8, 5))
    modes = ["Baseline", "DP", "FHE", "RAG"]
    times = [
        avg["baseline_latency_ms"],
        avg["dp_latency_ms"],
        avg["fhe_latency_ms"],
        avg["rag_latency_ms"],
    ]
    plt.bar(modes, times, color=["#4daf4a", "#377eb8", "#ff7f00", "#984ea3"])
    plt.yscale("log")
    plt.ylabel("Average Latency (ms, log scale)")
    plt.title("Query Latency Comparison (log scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_log.png"))
    plt.close()

    # ---- nDCG ----
    plt.figure(figsize=(8, 5))
    plt.bar(["RAG"], [avg["rag_ndcg"]], color="#984ea3")
    plt.ylim(0, 1.1)
    plt.ylabel("nDCG@10")
    plt.title("RAG Ranking Quality (nDCG@10)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ndcg_rag.png"))
    plt.close()

    # ---- Recall & Overlap ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    recall_means = [avg["dp_recall"], avg["fhe_recall"], avg["rag_recall"]]
    overlap_means = [avg["dp_overlap"], avg["fhe_overlap"], avg["rag_overlap"]]
    labels = ["DP", "FHE", "RAG"]

    ax[0].bar(labels, recall_means, color="#377eb8")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Recall@10 vs Baseline")

    ax[1].bar(labels, overlap_means, color="#ff7f00")
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Top-10 Overlap with Baseline")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "recall_overlap.png"))
    plt.close()

    # ---- Privacy–Utility scatter ----
    plt.figure(figsize=(7, 5))
    plt.scatter(df["dp_latency_ms"], df["dp_recall"], label="DP", color="#377eb8", s=80)
    plt.scatter(df["fhe_latency_ms"], df["fhe_recall"], label="FHE", color="#ff7f00", s=80)
    plt.scatter(df["rag_latency_ms"], df["rag_recall"], label="RAG", color="#984ea3", s=80)
    plt.xscale("log")
    plt.xlabel("Latency (ms, log scale)")
    plt.ylabel("Recall@10")
    plt.title("Privacy–Utility Tradeoff")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "privacy_utility.png"))
    plt.close()

    print(f"Plots saved under: {PLOTS_DIR}")


# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------
def main():
    data_path = os.path.join(PROJECT_ROOT, "src", "dataset", "medical_transcriptions.csv")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    index_paths = {
        "baseline": os.path.join(PROJECT_ROOT, "src", "faiss_baseline.faiss"),
        "dp": os.path.join(PROJECT_ROOT, "src", "faiss_dp.faiss"),
        "rag": os.path.join(PROJECT_ROOT, "src", "faiss_rag.faiss"),
    }

    df = benchmark_all(data_path, model_name, index_paths)
    generate_plots(df)


if __name__ == "__main__":
    main()
