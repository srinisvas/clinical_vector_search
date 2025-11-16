import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import faiss
import tenseal as ts
from sentence_transformers import SentenceTransformer

from pipeline.utils import normalize_rows, norm_vec


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

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


# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------

def run_timed(func, *args, **kwargs):
    start = time.time()
    res = func(*args, **kwargs)
    return res, (time.time() - start) * 1000.0


def recall_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(base)


def precision_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(other)


def accuracy_at_k(base, other):
    return sum(1 for a, b in zip(base, other) if a == b) / len(base)


def mrr(base, other):
    base_set = set(base)
    for rank, doc in enumerate(other, start=1):
        if doc in base_set:
            return 1.0 / rank
    return 0.0


# -------------------------------------------------------
# Extracting vectors from FlatIP FAISS Index (macOS-safe)
# -------------------------------------------------------

def extract_vectors_from_flat_index(index):
    """
    macOS-safe extraction of xb from FlatIP index.
    reconstruct_n cannot be used.
    """
    xb_ptr = index.get_xb()  # pointer → float array
    ntotal, dim = index.ntotal, index.d
    xb = faiss.rev_swig_ptr(xb_ptr, ntotal * dim)
    xb = xb.reshape(ntotal, dim).astype(np.float32)
    return normalize_rows(xb)


# -------------------------------------------------------
# Evaluation Pipeline
# -------------------------------------------------------

def benchmark_all(data_path, model_name, index_paths):

    print("\nInitializing environment...")
    faiss.omp_set_num_threads(1)

    print("Loading model...")
    model = SentenceTransformer(model_name)

    print("\nLoading FAISS indexes...")
    baseline_index = faiss.read_index(index_paths["baseline"])
    dp_index = faiss.read_index(index_paths["dp"])

    # Load vectors directly from baseline index
    print("Extracting vectors from baseline index...")
    xb = extract_vectors_from_flat_index(baseline_index)

    # Rebuild fresh HNSW (OPTIMIZED RAG)
    print("Building fresh HNSW RAG index...")
    dim = xb.shape[1]
    rag_index = faiss.IndexHNSWFlat(dim, 32)
    rag_index.hnsw.efConstruction = 200
    rag_index.hnsw.efSearch = 64
    rag_index.add(xb)

    results = []

    print("\nRunning evaluation...")
    for query in QUERIES:
        print(f"\n---- Query: {query}")

        qv = normalize_rows(model.encode([query]).astype(np.float32))

        # ------------------ BASELINE ------------------
        (_, lat_base) = run_timed(baseline_index.search, qv, 10)
        _, I_base = baseline_index.search(qv, 10)
        base_ids = I_base[0]

        # ------------------ DP MODE ------------------
        dp_dim = dp_index.d
        text_dim = qv.shape[1]

        if dp_dim == text_dim:
            qv_dp = qv
        else:
            attr_dim = dp_dim - text_dim
            z = np.zeros((1, attr_dim), np.float32)
            qv_dp = normalize_rows(np.hstack([qv * 0.7, z * 0.3]))

        (_, lat_dp) = run_timed(dp_index.search, qv_dp, 10)
        _, I_dp = dp_index.search(qv_dp, 10)
        dp_ids = I_dp[0]

        # ------------------ FHE MODE (synthetic) ------------------
        d_target = 128
        R = np.random.normal(0, 1 / np.sqrt(text_dim), (text_dim, d_target)).astype(np.float32)
        q_small = normalize_rows(qv @ R)
        q_small = norm_vec(q_small.ravel().astype(np.float32))

        fhe_subset = 200
        vecs_small = normalize_rows(np.random.randn(fhe_subset, d_target).astype(np.float32))

        # TenSEAL encrypted dot products
        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40

        enc_q = ts.ckks_vector(ctx, q_small.tolist())

        fhe_scores = []
        t0 = time.time()
        for v in vecs_small:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0

        fhe_ids = np.argsort(fhe_scores)[::-1][:10]

        # ------------------ OPTIMIZED RAG (HNSW) ------------------
        (_, lat_rag) = run_timed(rag_index.search, qv, 10)
        _, I_rag = rag_index.search(qv, 10)
        rag_ids = I_rag[0]

        # ------------------ Metrics ------------------
        results.append({
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
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(PROJECT_ROOT, "evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results → {out_csv}")
    return df


# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

def generate_plots(df):
    # ------------ LATENCY (log scale) ---------------
    plt.figure(figsize=(8, 5))
    modes = ["Baseline", "DP", "FHE", "RAG"]
    means = [
        df["baseline_latency_ms"].mean(),
        df["dp_latency_ms"].mean(),
        df["fhe_latency_ms"].mean(),
        df["rag_latency_ms"].mean(),
    ]
    plt.bar(modes, means)
    plt.yscale("log")
    plt.title("Latency (ms) - Log Scale")
    plt.ylabel("Latency (log ms)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_log.png"))
    plt.close()

    # ------------ 4 Metrics (side by side) ----------
    metric_map = {
        "recall": ["dp_recall", "fhe_recall", "rag_recall"],
        "precision": ["dp_precision", "fhe_precision", "rag_precision"],
        "accuracy": ["dp_accuracy", "fhe_accuracy", "rag_accuracy"],
        "mrr": ["dp_mrr", "fhe_mrr", "rag_mrr"],
    }

    for metric, cols in metric_map.items():
        plt.figure(figsize=(8, 5))
        for c in cols:
            plt.plot(df[c], marker="o", label=c.replace("_", " ").upper())
        plt.title(f"{metric.upper()} Comparison per Query")
        plt.xlabel("Query Index")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{metric}.png"))
        plt.close()

    print(f"Saved plots → {PLOTS_DIR}")


# -------------------------------------------------------
# Entry
# -------------------------------------------------------

def main():
    data_path = os.path.join(PROJECT_ROOT, "src", "dataset", "medical_transcriptions.csv")

    index_paths = {
        "baseline": os.path.join(PROJECT_ROOT, "src", "faiss_baseline.faiss"),
        "dp": os.path.join(PROJECT_ROOT, "src", "faiss_dp.faiss"),
        "rag": os.path.join(PROJECT_ROOT, "src", "faiss_rag.faiss"),  # optional only
    }

    df = benchmark_all(data_path, "sentence-transformers/all-MiniLM-L6-v2", index_paths)
    generate_plots(df)


if __name__ == "__main__":
    main()
