import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import faiss
import tenseal as ts
from sentence_transformers import SentenceTransformer

from pipeline.utils import normalize_rows, norm_vec


# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------

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


# ------------------------------------------------------
# METRIC HELPERS
# ------------------------------------------------------

def run_timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    ms = (time.time() - start) * 1000.0
    return out, ms


def recall_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(base)


def precision_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(other)


def accuracy_at_k(base, other):
    matches = sum(1 for a, b in zip(base, other) if a == b)
    return matches / len(base)


def mrr(base, other):
    base_set = set(base)
    for rank, did in enumerate(other, start=1):
        if did in base_set:
            return 1.0 / rank
    return 0.0


# ------------------------------------------------------
# SAFE VECTOR EXTRACTION FOR FHE
# ------------------------------------------------------

def get_safe_vectors_from_faiss(index, synthetic_min=500):
    """
    Try reconstructing vectors. If not supported, fall back to synthetic normalized vectors.
    Always macOS-safe (single-thread FAISS).
    """
    ntotal = index.ntotal
    dim = index.d

    # Always enforce single-thread FAISS on mac
    try:
        faiss.omp_set_num_threads(1)
    except:
        pass

    # Attempt reconstruction
    if hasattr(index, "reconstruct"):
        try:
            xb = np.zeros((ntotal, dim), dtype=np.float32)
            for i in range(ntotal):
                xb[i] = index.reconstruct(i)
            return normalize_rows(xb)
        except Exception as e:
            print(f"⚠️ Cannot reconstruct FAISS vectors: {e}")

    # Fallback: synthetic
    n = max(synthetic_min, ntotal if ntotal > 0 else synthetic_min)
    print(f"⚠️ Using synthetic vectors for FHE ({n} x {dim})...")
    xb = np.random.randn(n, dim).astype(np.float32)
    return normalize_rows(xb)


# ------------------------------------------------------
# MAIN EVALUATION
# ------------------------------------------------------

def benchmark_all(model_name, index_paths):
    print("\nLoading model...")
    model = SentenceTransformer(model_name)

    print("\nLoading FAISS indexes...")
    baseline_index = faiss.read_index(index_paths["baseline"])
    dp_index = faiss.read_index(index_paths["dp"])

    # Prepare vectors for FHE projection
    print("\nPreparing projection vectors for FHE...")
    vecs = get_safe_vectors_from_faiss(baseline_index)
    dim = vecs.shape[1]

    # Build a safe RAG HNSW index entirely in memory
    print("\nBuilding RAG HNSW index (macOS safe)...")
    rag_index = faiss.IndexHNSWFlat(dim, 32)
    rag_index.hnsw.efConstruction = 100
    rag_index.add(vecs)
    rag_index.hnsw.efSearch = 64

    results = []

    for query in QUERIES:
        print("\n-----------------------------------------------------")
        print(f"Query: {query}")

        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # ---------------- BASELINE ----------------
        (_, lat_base) = run_timed(baseline_index.search, qv, 10)
        _, I_base = baseline_index.search(qv, 10)
        base_ids = I_base[0]

        # ---------------- DP ----------------
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

        # ---------------- FHE (synthetic) ----------------
        d_target = 256
        R = np.random.normal(0, 1 / np.sqrt(dim), size=(dim, d_target)).astype(np.float32)

        qv_small = normalize_rows(qv @ R)
        qv_small = norm_vec(qv_small.ravel())

        fhe_subset = 200
        vecs_small = np.random.randn(fhe_subset, d_target).astype(np.float32)
        vecs_small = normalize_rows(vecs_small)

        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40

        enc_q = ts.ckks_vector(ctx, qv_small.tolist())

        t0 = time.time()
        fhe_scores = [enc_q.dot(v.tolist()).decrypt()[0] for v in vecs_small]
        lat_fhe = (time.time() - t0) * 1000.0
        fhe_ids = np.argsort(fhe_scores)[::-1][:10]

        # ---------------- RAG HNSW ----------------
        (_, lat_rag) = run_timed(rag_index.search, qv, 10)
        _, I_rag = rag_index.search(qv, 10)
        rag_ids = I_rag[0]

        # ---------------- METRICS ----------------
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
    print(f"\nSaved evaluation results to {out_csv}")

    return df


# ------------------------------------------------------
# PLOTTING
# ------------------------------------------------------

def generate_plots(df):
    plt.figure(figsize=(8, 5))
    avg = df.mean(numeric_only=True)

    modes = ["Baseline", "DP", "FHE", "RAG"]
    times = [
        avg["baseline_latency_ms"],
        avg["dp_latency_ms"],
        avg["fhe_latency_ms"],
        avg["rag_latency_ms"],
    ]

    plt.bar(modes, times)
    plt.ylabel("Latency (ms)")
    plt.title("Average Latency per Mode")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency.png"))
    plt.close()

    # Metric groups
    for metric in ["recall", "precision", "accuracy", "mrr"]:
        plt.figure(figsize=(8, 5))
        for mode in ["dp", "fhe", "rag"]:
            col = f"{mode}_{metric}"
            plt.plot(df[col], marker="o", label=mode.upper())
        plt.legend()
        plt.title(metric.upper())
        plt.xlabel("Query Index")
        plt.ylabel(metric.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{metric}.png"))
        plt.close()

    print(f"Saved all plots to {PLOTS_DIR}")


# ------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------

def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    index_paths = {
        "baseline": os.path.join(PROJECT_ROOT, "src", "faiss_baseline.faiss"),
        "dp": os.path.join(PROJECT_ROOT, "src", "faiss_dp.faiss"),
        "rag": os.path.join(PROJECT_ROOT, "src", "faiss_rag.faiss"),  # not used
    }

    df = benchmark_all(model_name, index_paths)
    generate_plots(df)


if __name__ == "__main__":
    main()
