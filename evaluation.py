import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss
import tenseal as ts

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
    "post-surgical infection management"
]

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------------------------------------
# Utility Functions
# -------------------------------------------------------

def run_timed(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    elapsed_ms = (time.time() - start) * 1000.0
    return result, elapsed_ms


def recall_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(base)


def precision_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(other)


def accuracy_at_k(base, other):
    matches = sum(1 for a, b in zip(base, other) if a == b)
    return matches / len(base)


def mrr(base, other):
    for rank, doc_id in enumerate(other, start=1):
        if doc_id in base:
            return 1.0 / rank
    return 0.0


# -------------------------------------------------------
# Evaluation Core
# -------------------------------------------------------

def benchmark_all(data_path, model_name, index_paths):
    print("\nLoading model...")
    model = SentenceTransformer(model_name)

    print("\nLoading embeddings parquet...")
    pdf = pd.read_parquet("embeddings.parquet")
    vecs = np.vstack(pdf["vec"].values).astype(np.float32)

    print("\nLoading Baseline FAISS index...")
    baseline_index = faiss.read_index(index_paths["baseline"])

    print("Loading DP index...")
    dp_index = faiss.read_index(index_paths["dp"])

    print("\nRebuilding RAG HNSW index in-memory (macOS safe)...")
    d = vecs.shape[1]
    rag_index = faiss.IndexHNSWFlat(d, 32)
    rag_index.hnsw.efConstruction = 200
    rag_index.add(vecs)
    rag_index.hnsw.efSearch = 128

    fhe_subset = 100  # small, fast, POC FHE

    results = []

    for query in QUERIES:
        print(f"\n----------------------------------------------")
        print(f"Evaluating query: '{query}'")

        # Encode query
        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # ------------------ Baseline ------------------
        (_, lat_base) = run_timed(baseline_index.search, qv, 10)
        _, I_base = baseline_index.search(qv, 10)
        base_ids = I_base[0]

        # ------------------ DP Mode ------------------
        dp_dim = dp_index.d
        text_dim = qv.shape[1]
        attr_dim = dp_dim - text_dim

        qv_attr = np.zeros((1, attr_dim), dtype=np.float32)
        qv_dp = normalize_rows(np.hstack([qv * 0.7, qv_attr * 0.3]))

        (_, lat_dp) = run_timed(dp_index.search, qv_dp, 10)
        _, I_dp = dp_index.search(qv_dp, 10)
        dp_ids = I_dp[0]

        # ------------------ FHE Mode (synthetic) ------------------
        d_target = 256
        R = np.random.normal(0, 1 / np.sqrt(qv.shape[1]), size=(qv.shape[1], d_target)).astype(np.float32)

        qv_small = normalize_rows(qv @ R)
        qv_small = norm_vec(qv_small.ravel().astype(np.float32))

        vecs_small = np.random.randn(fhe_subset, d_target).astype(np.float32)
        vecs_small = normalize_rows(vecs_small)

        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40

        enc_q = ts.ckks_vector(ctx, qv_small.tolist())

        fhe_scores = []
        t0 = time.time()
        for v in vecs_small:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0
        fhe_ids = np.argsort(fhe_scores)[::-1][:10]

        # ------------------ RAG (HNSW Accurate) ------------------
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
    df.to_csv("evaluation_results.csv", index=False)
    print("\nSaved evaluation_results.csv")
    return df


# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

def generate_plots(df):
    avg = df.mean(numeric_only=True)

    plt.figure(figsize=(8, 5))
    modes = ["Baseline", "DP", "FHE", "RAG"]
    times = [
        avg["baseline_latency_ms"],
        avg["dp_latency_ms"],
        avg["fhe_latency_ms"],
        avg["rag_latency_ms"]
    ]
    plt.bar(modes, times)
    plt.ylabel("Average Latency (ms)")
    plt.title("Query Latency Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_comparison.png"))
    plt.close()

    metrics = ["recall", "precision", "accuracy", "mrr"]
    for m in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(df[f"dp_{m}"], marker="o", label="DP")
        plt.plot(df[f"fhe_{m}"], marker="o", label="FHE")
        plt.plot(df[f"rag_{m}"], marker="o", label="RAG")
        plt.legend()
        plt.title(f"{m.upper()} Comparison")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{m}.png"))
        plt.close()

    print("Saved plots to /plots/")


# -------------------------------------------------------
# Entry
# -------------------------------------------------------

def main():
    data_path = "./data/medical_transcriptions.csv"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    index_paths = {
        "baseline": "./faiss_mtsamples_baseline.faiss",
        "dp": "./faiss_mtsamples_dp.faiss",
        "rag": "./faiss_mtsamples_rag_hnsw.faiss"  # not used directly
    }

    df = benchmark_all(data_path, model_name, index_paths)
    generate_plots(df)


if __name__ == "__main__":
    main()
