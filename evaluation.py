import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import faiss
import matplotlib.pyplot as plt
import tenseal as ts

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pipeline.utils import normalize_rows, norm_vec


# -------------------------------------------------------
# Config
# -------------------------------------------------------

QUERIES = [
    "post operative knee arthroscopy pain management",
    "chest pain with ECG changes",
    "abdominal pain after appendectomy",
    "shortness of breath with asthma history",
    "lumbar spine MRI findings",
    "arthroscopic shoulder repair recovery",
    "ECG abnormalities in heart attack",
    "knee joint effusion",
    "diabetic foot ulcer treatment",
    "post surgical infection management",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# -------------------------------------------------------
# Utils
# -------------------------------------------------------

def timed(f, *a, **kw):
    t0 = time.time()
    out = f(*a, **kw)
    dt = (time.time() - t0) * 1000.0
    return out, dt

def recall_at_k(base, other):
    base = list(base)
    if len(base) == 0:
        return 0.0
    return len(set(base).intersection(set(other))) / len(base)

def precision_at_k(base, other):
    other = list(other)
    if len(other) == 0:
        return 0.0
    return len(set(base).intersection(set(other))) / len(other)

def accuracy_at_k(base, other):
    base = list(base)
    other = list(other)
    matches = sum(1 for x, y in zip(base, other) if x == y)
    return matches / len(base)

def mrr(base, other):
    base = set(base)
    for rank, docid in enumerate(other, start=1):
        if docid in base:
            return 1.0 / rank
    return 0.0


# -------------------------------------------------------
# Core evaluation
# -------------------------------------------------------

def benchmark_all(data_path, model_name, index_paths):
    print("Loading model...")
    model = SentenceTransformer(model_name)

    print("Loading FAISS baseline (FlatIP) index...")
    baseline_index = faiss.read_index(index_paths["baseline"])
    d = baseline_index.d

    print("Loading DP index...")
    dp_index = faiss.read_index(index_paths["dp"])

    print("Extracting baseline embeddings from FlatIP index...")
    xb = faiss.vector_to_array(baseline_index.get_xb()).reshape(baseline_index.ntotal, baseline_index.d)
    xb = normalize_rows(xb)

    print("Building Optimized RAG HNSW index...")
    rag_index = faiss.IndexHNSWFlat(d, 32)
    rag_index.hnsw.efConstruction = 200
    rag_index.add(xb)
    rag_index.hnsw.efSearch = 64

    print("Loading raw text for BM25...")
    df_raw = pd.read_csv(data_path)
    corpus = df_raw["transcription"].astype(str).tolist()
    tokenized = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    results = []

    # FHE subset of real embeddings
    fhe_subset = min(150, xb.shape[0])
    xb_small = xb[:fhe_subset]

    for query in QUERIES:
        print("\nProcessing query:", query)

        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # ----------------------------------------------------
        # Baseline RAG (FlatIP)
        # ----------------------------------------------------
        (_, base_lat) = timed(baseline_index.search, qv, 10)
        _, I_base = baseline_index.search(qv, 10)
        base_ids = I_base[0]

        # ----------------------------------------------------
        # DP
        # ----------------------------------------------------
        dp_dim = dp_index.d
        text_dim = qv.shape[1]
        if dp_dim > text_dim:
            attr_dim = dp_dim - text_dim
            qv_attr = np.zeros((1, attr_dim), dtype=np.float32)
            qv_dp = normalize_rows(np.hstack([qv * 0.7, qv_attr * 0.3]))
        else:
            qv_dp = qv

        (_, dp_lat) = timed(dp_index.search, qv_dp, 10)
        _, I_dp = dp_index.search(qv_dp, 10)
        dp_ids = I_dp[0]

        # ----------------------------------------------------
        # Optimized RAG breakdown
        # ----------------------------------------------------

        # 1. Vector search HNSW
        (_, rag_vec_lat) = timed(rag_index.search, qv, 50)
        _, temp_ids = rag_index.search(qv, 50)
        vector_candidates = list(temp_ids[0])

        # 2. BM25 hybrid
        (_, rag_bm25_lat) = timed(bm25.get_top_n, query.split(), corpus, n=50)
        bm25_indices = [corpus.index(doc) for doc in bm25.get_top_n(query.split(), corpus, n=50)]
        combined = set(vector_candidates).union(set(bm25_indices))

        # 3. MMR reranking
        cand_vecs = xb[list(combined)]
        query_vec = qv.ravel()

        (_, rag_mmr_lat), final_ids = timed(
            lambda: mmr_rerank(query_vec, cand_vecs, list(combined), k=10),
        )

        rag_total = rag_vec_lat + rag_bm25_lat + rag_mmr_lat

        # ----------------------------------------------------
        # FHE using real embeddings subset
        # ----------------------------------------------------
        d_target = 128
        R = np.random.normal(0, 1 / np.sqrt(d), size=(d, d_target)).astype(np.float32)

        q_small = normalize_rows(qv @ R)
        q_small = norm_vec(q_small.ravel().astype(np.float32))

        emb_small = normalize_rows(xb_small @ R)

        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.generate_galois_keys()
        ctx.global_scale = 2 ** 40

        enc_q = ts.ckks_vector(ctx, q_small.tolist())

        fhe_scores = []
        t0 = time.time()
        for v in emb_small:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        fhe_lat = (time.time() - t0) * 1000.0

        fhe_ids = np.argsort(fhe_scores)[::-1][:10]

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        results.append({
            "query": query,
            "baseline_latency": base_lat,
            "dp_latency": dp_lat,
            "rag_vector_latency": rag_vec_lat,
            "rag_bm25_latency": rag_bm25_lat,
            "rag_mmr_latency": rag_mmr_lat,
            "rag_total_latency": rag_total,
            "fhe_latency": fhe_lat,
            "dp_recall": recall_at_k(base_ids, dp_ids),
            "fhe_recall": recall_at_k(base_ids, fhe_ids),
            "rag_recall": recall_at_k(base_ids, final_ids),
            "dp_precision": precision_at_k(base_ids, dp_ids),
            "fhe_precision": precision_at_k(base_ids, fhe_ids),
            "rag_precision": precision_at_k(base_ids, final_ids),
            "dp_accuracy": accuracy_at_k(base_ids, dp_ids),
            "fhe_accuracy": accuracy_at_k(base_ids, fhe_ids),
            "rag_accuracy": accuracy_at_k(base_ids, final_ids),
            "dp_mrr": mrr(base_ids, dp_ids),
            "fhe_mrr": mrr(base_ids, fhe_ids),
            "rag_mrr": mrr(base_ids, final_ids),
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(PROJECT_ROOT, "evaluation_results.csv"), index=False)
    return df


# -------------------------------------------------------
# Plotting
# -------------------------------------------------------

def generate_plots(df):
    # Log scale latency plot
    plt.figure(figsize=(10, 6))
    modes = [
        "baseline_latency", "dp_latency",
        "rag_vector_latency", "rag_bm25_latency", "rag_mmr_latency",
        "rag_total_latency", "fhe_latency"
    ]
    avg = df.mean(numeric_only=True)[modes]

    plt.bar(modes, avg)
    plt.yscale("log")
    plt.title("Latency Comparison (log scale)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_logscale.png"))
    plt.close()

    # Quality metrics
    metrics = ["recall", "precision", "accuracy", "mrr"]
    for m in metrics:
        plt.figure(figsize=(8, 5))
        for mode in ["dp", "fhe", "rag"]:
            plt.plot(df[f"{mode}_{m}"], label=mode.upper(), marker="o")
        plt.title(f"{m.upper()} across queries")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{m}.png"))
        plt.close()


# -------------------------------------------------------
# Entry
# -------------------------------------------------------

def main():
    data_path = os.path.join(PROJECT_ROOT, "src", "dataset", "medical_transcriptions.csv")
    index_paths = {
        "baseline": os.path.join(PROJECT_ROOT, "src", "faiss_baseline.faiss"),
        "dp": os.path.join(PROJECT_ROOT, "src", "faiss_dp.faiss"),
        "rag": os.path.join(PROJECT_ROOT, "src", "faiss_rag.faiss"),
    }
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    df = benchmark_all(data_path, model_name, index_paths)
    generate_plots(df)
    print("Saved all plots and evaluation.")


if __name__ == "__main__":
    main()
