import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
import faiss

from pipeline.pipeline import search_faiss
from pipeline.utils import normalize_rows, norm_vec
import tenseal as ts


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
    res = func(*args, **kwargs)
    return res, (time.time() - start) * 1000.0


def recall_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(base)


def precision_at_k(base, other):
    return len(set(base).intersection(set(other))) / len(other)


def accuracy_at_k(base, other):
    # Hit rate: proportion of exact matches
    matches = sum(1 for a, b in zip(base, other) if a == b)
    return matches / len(base)


def mrr(base, other):
    # Rank position of first baseline item in other ranking
    for rank, doc_id in enumerate(other, start=1):
        if doc_id in base:
            return 1.0 / rank
    return 0.0


# -------------------------------------------------------
# Evaluation Core
# -------------------------------------------------------

def benchmark_all(data_path, model_name, index_paths):
    model = SentenceTransformer(model_name)
    results = []

    # Load prebuilt indexes
    baseline_index = faiss.read_index(index_paths["baseline"])
    dp_index = faiss.read_index(index_paths["dp"])
    rag_index = faiss.read_index(index_paths["rag"])

    fhe_subset = 200

    for query in QUERIES:
        print(f"\nEvaluating query: '{query}'")
        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # ------------------ Baseline ------------------
        (_, lat_base) = run_timed(search_faiss, baseline_index, qv, 10)
        D_base, I_base = search_faiss(baseline_index, qv, 10)
        base_ids = I_base[0]

        # ------------------ DP Mode ------------------
        d = dp_index.d
        text_dim = qv.shape[1]
        attr_dim = d - text_dim
        qv_attr = np.zeros((1, attr_dim), dtype=np.float32)
        qv_combined = np.hstack([qv * 0.7, qv_attr * 0.3])
        qv_combined = normalize_rows(qv_combined)

        (_, lat_dp) = run_timed(search_faiss, dp_index, qv_combined, 10)
        D_dp, I_dp = search_faiss(dp_index, qv_combined, 10)
        dp_ids = I_dp[0]

        # ------------------ FHE Mode ------------------
        d_target = 256
        rng = np.random.default_rng(1234)
        R = rng.normal(0, 1 / np.sqrt(qv.shape[1]), size=(qv.shape[1], d_target)).astype(np.float32)
        qv_small = normalize_rows(qv @ R)
        qv_small = norm_vec(qv_small.ravel().astype(np.float32))

        vecs = np.random.randn(fhe_subset, qv_small.shape[0]).astype(np.float32)
        vecs = normalize_rows(vecs)

        ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                         poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40
        enc_q = ts.ckks_vector(ctx, qv_small.tolist())

        fhe_scores = []
        t0 = time.time()
        for v in vecs:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0

        # We have no faiss index so use synthetic top-k
        fhe_ids = np.argsort(fhe_scores)[::-1][:10]

        # ------------------ RAG (HNSW) ------------------
        (_, lat_rag) = run_timed(search_faiss, rag_index, qv, 10)
        D_rag, I_rag = search_faiss(rag_index, qv, 10)
        rag_ids = I_rag[0]

        # -------------------------------------------------------
        # Compute final 5 metrics
        # -------------------------------------------------------

        results.append({
            "query": query,

            # Latency
            "baseline_latency_ms": lat_base,
            "dp_latency_ms": lat_dp,
            "fhe_latency_ms": lat_fhe,
            "rag_latency_ms": lat_rag,

            # Recall
            "dp_recall": recall_at_k(base_ids, dp_ids),
            "fhe_recall": recall_at_k(base_ids, fhe_ids),
            "rag_recall": recall_at_k(base_ids, rag_ids),

            # Precision
            "dp_precision": precision_at_k(base_ids, dp_ids),
            "fhe_precision": precision_at_k(base_ids, fhe_ids),
            "rag_precision": precision_at_k(base_ids, rag_ids),

            # Accuracy / Hit Rate
            "dp_accuracy": accuracy_at_k(base_ids, dp_ids),
            "fhe_accuracy": accuracy_at_k(base_ids, fhe_ids),
            "rag_accuracy": accuracy_at_k(base_ids, rag_ids),

            # MRR
            "dp_mrr": mrr(base_ids, dp_ids),
            "fhe_mrr": mrr(base_ids, fhe_ids),
            "rag_mrr": mrr(base_ids, rag_ids),
        })

    df = pd.DataFrame(results)
    df.to_csv("evaluation_results.csv", index=False)
    print("Saved: evaluation_results.csv")
    return df


# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

def generate_plots(df):
    avg = df.mean(numeric_only=True)

    # Latency bar chart
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

    # Recall comparison
    plt.figure(figsize=(8, 5))
    plt.plot(df["dp_recall"], label="DP", marker="o")
    plt.plot(df["fhe_recall"], label="FHE", marker="o")
    plt.plot(df["rag_recall"], label="RAG", marker="o")
    plt.title("Recall@K vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "recall.png"))
    plt.close()

    # Precision comparison
    plt.figure(figsize=(8, 5))
    plt.plot(df["dp_precision"], label="DP", marker="o")
    plt.plot(df["fhe_precision"], label="FHE", marker="o")
    plt.plot(df["rag_precision"], label="RAG", marker="o")
    plt.title("Precision@K vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "precision.png"))
    plt.close()

    # Accuracy comparison
    plt.figure(figsize=(8, 5))
    plt.plot(df["dp_accuracy"], label="DP", marker="o")
    plt.plot(df["fhe_accuracy"], label="FHE", marker="o")
    plt.plot(df["rag_accuracy"], label="RAG", marker="o")
    plt.title("Accuracy@K (Hit Rate)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "accuracy.png"))
    plt.close()

    # MRR comparison
    plt.figure(figsize=(8, 5))
    plt.plot(df["dp_mrr"], label="DP", marker="o")
    plt.plot(df["fhe_mrr"], label="FHE", marker="o")
    plt.plot(df["rag_mrr"], label="RAG", marker="o")
    plt.title("MRR vs Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "mrr.png"))
    plt.close()

    print("Plots saved.")


# -------------------------------------------------------
# Entry
# -------------------------------------------------------

def main():
    data_path = "./data/medical_transcriptions.csv"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    index_paths = {
        "baseline": "./faiss_mtsamples_baseline.faiss",
        "dp": "./faiss_mtsamples_dp.faiss",
        "rag": "./faiss_mtsamples_rag.faiss",
    }

    df = benchmark_all(data_path, model_name, index_paths)
    generate_plots(df)


if __name__ == "__main__":
    main()