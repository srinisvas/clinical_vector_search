# ==========================================================
# Clinical Vector Search Evaluation (macOS-safe)
# Baseline (FlatIP) vs DP vs Hybrid-RAG vs synthetic FHE
# Metrics: Latency, Semantic Drift, Rank Correlation,
#          Diversity, DP Embedding Drift
# ==========================================================

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

import faiss
import tenseal as ts

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


# Fix Mac FAISS threading (avoid segfaults)
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass


# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def normalize_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

def norm_vec(v):
    return v / (np.linalg.norm(v) + 1e-9)

def run_timed(func, *args, **kw):
    t0 = time.time()
    out = func(*args, **kw)
    return out, (time.time() - t0) * 1000.0


# 1. Semantic drift: average cosine similarity
def semantic_drift(baseline_vecs, method_vecs):
    if len(baseline_vecs) == 0:
        return 0.0
    sims = cosine_similarity(baseline_vecs, method_vecs)
    return float(np.mean(sims))


# 2. Rank correlation: Spearman
def rank_corr(base_ids, other_ids):
    if len(base_ids) < 2:
        return 1.0
    corr, _ = spearmanr(base_ids, other_ids)
    return 0.0 if corr != corr else float(corr)  # handle NaN


# 3. Diversity: avg pairwise distance within top-k
def diversity(vecs):
    if len(vecs) <= 1:
        return 0.0
    sims = cosine_similarity(vecs)
    triu = sims[np.triu_indices_from(sims, k=1)]
    return float(np.mean(1 - triu))  # distance = (1 - cosine)


# ----------------------------------------------------------
# Main evaluation
# ----------------------------------------------------------

def benchmark_all():

    # Load dataset text (BM25 needed full text)
    df_txt = pd.read_csv(os.path.join(PROJECT_ROOT, "src", "dataset", "medical_transcriptions.csv"))
    df_txt["text"] = df_txt["transcription"].astype(str)
    corpus = df_txt["text"].fillna("").tolist()

    tokenized = [t.lower().split() for t in corpus]
    bm25 = BM25Okapi(tokenized)

    # Sentence model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # Load FAISS indexes
    baseline_path = os.path.join(PROJECT_ROOT, "src", "faiss_baseline.faiss")
    dp_path = os.path.join(PROJECT_ROOT, "src", "faiss_dp.faiss")
    rag_path = os.path.join(PROJECT_ROOT, "src", "faiss_rag.faiss")

    print("Loading FAISS indexes...")
    baseline_index = faiss.read_index(baseline_path)
    dp_index = faiss.read_index(dp_path)

    # Build RAG HNSW index from baseline vectors
    print("Rebuilding HNSW for RAG...")
    xb_ptr = baseline_index.get_xb()
    d = baseline_index.d
    n = baseline_index.ntotal
    xb = faiss.rev_swig_ptr(xb_ptr, n * d).reshape(n, d)
    xb = normalize_rows(xb)

    rag_index = faiss.IndexHNSWFlat(d, 32)
    rag_index.hnsw.efConstruction = 100
    rag_index.add(xb)
    rag_index.hnsw.efSearch = 128

    results = []

    for query in QUERIES:
        print(f"\nEvaluating: {query}")

        # Encode query
        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # --------------------------------------------------
        # BASELINE
        # --------------------------------------------------
        (_, lat_base) = run_timed(baseline_index.search, qv, 10)
        _, I_base = baseline_index.search(qv, 10)
        base_ids = I_base[0]
        base_vecs = xb[base_ids]

        # --------------------------------------------------
        # DP
        # --------------------------------------------------
        dp_d = dp_index.d
        if dp_d > d:
            attr_dim = dp_d - d
            qv_dp = np.hstack([qv * 0.7, np.zeros((1, attr_dim)) * 0.3])
            qv_dp = normalize_rows(qv_dp)
        else:
            qv_dp = qv

        (_, lat_dp) = run_timed(dp_index.search, qv_dp, 10)
        _, I_dp = dp_index.search(qv_dp, 10)
        dp_ids = I_dp[0]
        dp_vecs = xb[dp_ids[:10]] if len(dp_ids) >= 10 else xb[dp_ids]

        # DP embedding drift
        dp_drift = float(cosine_similarity(qv, qv_dp)[0][0])

        # --------------------------------------------------
        # RAG Hybrid (HNSW + BM25 + MMR)
        # --------------------------------------------------
        (_, lat_rag) = run_timed(rag_index.search, qv, 50)
        _, I_hnsw = rag_index.search(qv, 50)
        h_candidates = set(I_hnsw[0])

        bm_ids = bm25.get_top_n(query.lower().split(), list(range(n)), n=50)
        rag_cand = list(set(bm_ids) | h_candidates)

        # MMR
        cand_vecs = xb[rag_cand]
        qv_r = qv.ravel()

        scores = cand_vecs @ qv_r
        selected = []
        remaining = list(range(len(rag_cand)))

        while len(selected) < 10 and remaining:
            if len(selected) == 0:
                best = int(np.argmax(scores))
                selected.append(best)
                remaining.remove(best)
            else:
                sel_vecs = cand_vecs[selected]
                sim_to_selected = cosine_similarity(cand_vecs[remaining], sel_vecs).max(axis=1)
                mmr_score = 0.7 * scores[remaining] - 0.3 * sim_to_selected
                best_local = int(np.argmax(mmr_score))
                best = remaining[best_local]
                selected.append(best)
                remaining.remove(best)

        rag_ids = [rag_cand[i] for i in selected]
        rag_vecs = xb[rag_ids]

        # --------------------------------------------------
        # Synthetic FHE
        # --------------------------------------------------
        fhe_subset = 200
        d_target = 256
        R = np.random.normal(0, 1 / np.sqrt(d), (d, d_target)).astype(np.float32)

        qv_small = normalize_rows(qv @ R)
        qv_small = norm_vec(qv_small.ravel())

        vecs_small = normalize_rows(np.random.randn(fhe_subset, d_target).astype(np.float32))

        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.generate_galois_keys()
        ctx.global_scale = 2**40

        enc_q = ts.ckks_vector(ctx, qv_small.tolist())

        fhe_scores = []
        t0 = time.time()
        for v in vecs_small:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0

        fhe_ids = np.argsort(fhe_scores)[::-1][:10]
        fhe_vecs = vecs_small[fhe_ids]

        # --------------------------------------------------
        # Compute metrics
        # --------------------------------------------------
        results.append({
            "query": query,
            "baseline_latency": lat_base,
            "dp_latency": lat_dp,
            "rag_latency": lat_rag,
            "fhe_latency": lat_fhe,

            "dp_semantic_drift": semantic_drift(base_vecs, dp_vecs),
            "rag_semantic_drift": semantic_drift(base_vecs, rag_vecs),
            "fhe_semantic_drift": semantic_drift(base_vecs, fhe_vecs),

            "dp_rank_corr": rank_corr(base_ids, dp_ids),
            "rag_rank_corr": rank_corr(base_ids, rag_ids),
            "fhe_rank_corr": 0.0,  # synthetic

            "dp_diversity": diversity(dp_vecs),
            "rag_diversity": diversity(rag_vecs),
            "fhe_diversity": diversity(fhe_vecs),

            "dp_embedding_drift": dp_drift
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(PROJECT_ROOT, "evaluation_results.csv"), index=False)
    print("\nSaved evaluation_results.csv")
    return df


# ----------------------------------------------------------
# Plotting
# ----------------------------------------------------------

def generate_plots(df):

    # -----------------------
    # Latency (log scale)
    # -----------------------
    plt.figure(figsize=(8, 5))
    plt.yscale("log")
    modes = ["baseline_latency", "dp_latency", "rag_latency", "fhe_latency"]
    values = [df[modes].mean()[m] for m in modes]

    plt.bar(["Baseline", "DP", "RAG", "FHE"], values)
    plt.ylabel("Latency (log scale, ms)")
    plt.title("Latency Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency.png"))
    plt.close()

    # Metric curves
    metrics = [
        ("semantic_drift", ["dp_semantic_drift", "rag_semantic_drift", "fhe_semantic_drift"]),
        ("rank_corr", ["dp_rank_corr", "rag_rank_corr", "fhe_rank_corr"]),
        ("diversity", ["dp_diversity", "rag_diversity", "fhe_diversity"]),
        ("dp_embedding_drift", ["dp_embedding_drift"])
    ]

    for title, cols in metrics:
        plt.figure(figsize=(8, 5))
        for col in cols:
            plt.plot(df[col], marker="o", label=col.upper())
        plt.legend()
        plt.title(title.upper())
        plt.xlabel("Query index")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{title}.png"))
        plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


# ----------------------------------------------------------
# Entry
# ----------------------------------------------------------

def main():
    df = benchmark_all()
    generate_plots(df)


if __name__ == "__main__":
    main()
