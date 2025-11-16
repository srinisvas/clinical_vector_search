import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    "post-surgical infection management",
]

# Resolve project root from this file: .../src/pipeline/evaluation.py -> .../
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
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
    base_set = list(base)
    other_set = list(other)
    if not base_set:
        return 0.0
    return len(set(base_set).intersection(set(other_set))) / len(base_set)


def precision_at_k(base, other):
    base_set = list(base)
    other_set = list(other)
    if not other_set:
        return 0.0
    return len(set(base_set).intersection(set(other_set))) / len(other_set)


def accuracy_at_k(base, other):
    base_list = list(base)
    other_list = list(other)
    if not base_list:
        return 0.0
    matches = sum(1 for a, b in zip(base_list, other_list) if a == b)
    return matches / len(base_list)


def mrr(base, other):
    base_set = set(base)
    for rank, doc_id in enumerate(other, start=1):
        if doc_id in base_set:
            return 1.0 / rank
    return 0.0


# -------------------------------------------------------
# Evaluation Core
# -------------------------------------------------------

def benchmark_all(data_path, model_name, index_paths):
    """Run baseline, DP, FHE (synthetic), and RAG-style HNSW evaluation.

    Assumptions / contracts:
    - `embeddings.parquet` exists at `src/mt_samples_embeddings.parquet` or
      `src/embeddings.parquet` and contains a column `vec` of embedding vectors.
    - `index_paths` contains FAISS index paths for baseline and DP.
    - Designed to be run from project root via `python -m src.pipeline.evaluation`.
    - FAISS is used in CPU-only mode; threading is restricted on macOS.
    """

    # Make FAISS safer on macOS: restrict to 1 thread to avoid segfaults
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
        pass

    print("\nLoading model...")
    model = SentenceTransformer(model_name)

    # ---- Load embeddings parquet (robust path resolution) ----
    print("\nLoading embeddings parquet...")
    # Prefer the project `src/mt_samples_embeddings.parquet` if present
    cand_paths = [
        os.path.join(PROJECT_ROOT, "src", "mt_samples_embeddings.parquet"),
        os.path.join(PROJECT_ROOT, "src", "embeddings.parquet"),
    ]
    pdf = None
    for p in cand_paths:
        if os.path.exists(p):
            print(f"Using embeddings file: {p}")
            pdf = pd.read_parquet(p)
            break
    if pdf is None:
        raise FileNotFoundError(
            "Could not find embeddings parquet. Checked: " + ", ".join(cand_paths)
        )

    if "vec" not in pdf.columns:
        raise ValueError("Embeddings parquet must contain a 'vec' column with vectors.")

    vecs = np.vstack(pdf["vec"].values).astype(np.float32)
    vecs = normalize_rows(vecs)

    # ---- Load FAISS indexes ----
    print("\nLoading Baseline FAISS index...")
    baseline_index = faiss.read_index(index_paths["baseline"])

    print("Loading DP index...")
    dp_index = faiss.read_index(index_paths["dp"])

    # ---- Build in-memory HNSW index for RAG-style eval ----
    print("\nRebuilding RAG HNSW index in-memory (macOS safe)...")
    d = vecs.shape[1]
    rag_index = faiss.IndexHNSWFlat(d, 32)
    rag_index.hnsw.efConstruction = 100
    rag_index.add(vecs)
    rag_index.hnsw.efSearch = 64

    # For FHE, we evaluate a small synthetic subset to keep runtime reasonable.
    fhe_subset = min(200, vecs.shape[0])

    results = []

    for query in QUERIES:
        print("\n----------------------------------------------")
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
        if dp_dim <= text_dim:
            # Fallback: treat DP index like baseline if shapes don't match
            qv_dp = qv
        else:
            attr_dim = dp_dim - text_dim
            qv_attr = np.zeros((1, attr_dim), dtype=np.float32)
            qv_dp = normalize_rows(np.hstack([qv * 0.7, qv_attr * 0.3]))

        (_, lat_dp) = run_timed(dp_index.search, qv_dp, 10)
        _, I_dp = dp_index.search(qv_dp, 10)
        dp_ids = I_dp[0]

        # ------------------ FHE Mode (synthetic, CPU-safe) ------------------
        # Dimensionality reduction for FHE demo
        d_target = 256
        R = np.random.normal(
            0,
            1 / np.sqrt(qv.shape[1]),
            size=(qv.shape[1], d_target),
        ).astype(np.float32)

        qv_small = normalize_rows(qv @ R)
        qv_small = norm_vec(qv_small.ravel().astype(np.float32))

        # Use synthetic normalized vectors (no need to encrypt the entire corpus)
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

        fhe_scores = []
        t0 = time.time()
        for v in vecs_small:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0
        fhe_ids = np.argsort(fhe_scores)[::-1][:10]

        # ------------------ RAG (HNSW) ------------------
        (_, lat_rag) = run_timed(rag_index.search, qv, 10)
        _, I_rag = rag_index.search(qv, 10)
        rag_ids = I_rag[0]

        # ------------------ Metrics ------------------
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
            }
        )

    df = pd.DataFrame(results)
    out_csv = os.path.join(PROJECT_ROOT, "evaluation_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved evaluation results to {out_csv}")
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
        avg.get("baseline_latency_ms", 0.0),
        avg.get("dp_latency_ms", 0.0),
        avg.get("fhe_latency_ms", 0.0),
        avg.get("rag_latency_ms", 0.0),
    ]
    plt.bar(modes, times)
    plt.ylabel("Average Latency (ms)")
    plt.title("Query Latency Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_comparison.png"))
    plt.close()

    # Metric curves per query
    metrics = ["recall", "precision", "accuracy", "mrr"]
    for m in metrics:
        plt.figure(figsize=(8, 5))
        for mode in ["dp", "fhe", "rag"]:
            col = f"{mode}_{m}"
            if col in df.columns:
                plt.plot(df[col], marker="o", label=mode.upper())
        plt.legend()
        plt.title(f"{m.upper()} Comparison")
        plt.xlabel("Query index")
        plt.ylabel(m.upper())
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{m}.png"))
        plt.close()

    print(f"Saved plots to {PLOTS_DIR}")


# -------------------------------------------------------
# Entry
# -------------------------------------------------------

def main():
    data_path = os.path.join(PROJECT_ROOT, "src", "dataset", "medical_transcriptions.csv")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    index_paths = {
        "baseline": os.path.join(PROJECT_ROOT, "src", "faiss_baseline.faiss"),
        "dp": os.path.join(PROJECT_ROOT, "src", "faiss_dp.faiss"),
        # RAG index is built in-memory from embeddings; file path kept for compatibility
        "rag": os.path.join(PROJECT_ROOT, "src", "faiss_rag.faiss"),
    }

    df = benchmark_all(data_path, model_name, index_paths)
    generate_plots(df)


if __name__ == "__main__":
    main()
