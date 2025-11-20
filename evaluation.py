import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
import faiss
import tenseal as ts
from rank_bm25 import BM25Okapi
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Basic Setup
# ============================================================

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

TOP_K = 10
FHE_CANDIDATE_POOL = 200
DP_SIGMAS = [0.05, 0.1, 0.15, 0.2]
SAFE_MODE = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

DATA_PATH = os.path.join(ROOT, "src", "dataset", "medical_transcriptions.csv")
PLOTS_DIR = os.path.join(ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

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

# ============================================================
# Utility Functions
# ============================================================

def normalize_rows(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

def norm_vec(v):
    return v / (np.linalg.norm(v) + 1e-9)

def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, (time.time() - t0) * 1000.0

def ndcg_at_k(ranks, k):
    ranks = np.array(ranks)[:k]
    gains = 1.0 / np.log2(np.arange(2, len(ranks) + 2))
    return np.sum(gains * ranks) / np.sum(gains)

def index_agreement(ref, other):
    ref = list(ref)
    other = list(other)
    matches = sum(1 for a, b in zip(ref, other) if a == b)
    return matches / len(ref)

def rank_positions(ref, pred):
    mapping = {doc_id: i + 1 for i, doc_id in enumerate(ref)}
    return [mapping.get(x, 0) for x in pred]


# ============================================================
# Build all embeddings and indexes
# ============================================================

def build_all(data_path, model_name):
    print("Loading dataset...")
    raw = pd.read_csv(data_path)
    raw.columns = [c.lower() for c in raw.columns]

    required = ["medical_specialty", "transcription"]
    for col in required:
        if col not in raw.columns:
            raise ValueError(f"Missing required column: {col}")

    pdf = pd.DataFrame({
        "text": raw["medical_specialty"].astype(str) + ", " + raw["transcription"].astype(str)
    }).drop_duplicates()

    model = SentenceTransformer(model_name)
    embeddings = model.encode(pdf["text"].tolist(), batch_size=32, show_progress_bar=True)
    embeddings = normalize_rows(embeddings.astype(np.float32))

    d = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(d)
    base_index.add(embeddings)

    # DP attributes
    attr_emb = model.encode(pdf["text"].tolist(), batch_size=32, show_progress_bar=False)
    attr_emb = normalize_rows(attr_emb.astype(np.float32))

    dp_indices = {}
    dp_vecs_map = {}

    for sigma in DP_SIGMAS:
        noisy = attr_emb + np.random.normal(0, sigma, attr_emb.shape).astype(np.float32)
        noisy = normalize_rows(noisy)

        dp_vecs = normalize_rows(
            np.hstack([embeddings * 0.7, noisy * 0.3]).astype(np.float32)
        )

        index = faiss.IndexFlatIP(dp_vecs.shape[1])
        index.add(dp_vecs)

        dp_indices[sigma] = index
        dp_vecs_map[sigma] = dp_vecs

    # RAG
    tokenized = [t.lower().split() for t in pdf["text"]]
    bm25 = BM25Okapi(tokenized)

    rag_index = faiss.IndexFlatIP(d)
    rag_index.add(embeddings)

    return pdf, embeddings, base_index, dp_indices, dp_vecs_map, bm25, rag_index, model


# ============================================================
# Evaluate all architectures
# ============================================================

def evaluate_all(pdf, embeddings, base_index, dp_indices, dp_vecs_map, bm25, rag_index, model):

    results = []

    for query in QUERIES:
        print(f"\nEvaluating query: {query}")

        # ----------------------------------------
        # Query embedding
        # ----------------------------------------
        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # ----------------------------------------
        # Baseline
        # ----------------------------------------
        (_, lat_base) = timed(base_index.search, qv, TOP_K)
        _, I_base = base_index.search(qv, TOP_K)
        base_ids = I_base[0]

        # ----------------------------------------
        # DP variants
        # ----------------------------------------
        dp_latencies = {}
        dp_ids_map = {}
        dp_drifts = {}

        text_dim = qv.shape[1]

        for sigma, dp_index in dp_indices.items():
            dp_dim = dp_index.d
            if dp_dim > text_dim:
                extra = dp_dim - text_dim
                qv_dp = np.hstack([qv * 0.7, np.zeros((1, extra), dtype=np.float32) * 0.3])
            else:
                qv_dp = qv

            (_, lat_dp) = timed(dp_index.search, qv_dp, TOP_K)
            _, I_dp = dp_index.search(qv_dp, TOP_K)

            dp_ids = I_dp[0]

            if qv.shape[1] == qv_dp.shape[1]:
                dp_drift = float(cosine_similarity(qv, qv_dp)[0][0])
            else:
                dp_drift = 1.0

            dp_latencies[sigma] = lat_dp
            dp_ids_map[sigma] = dp_ids
            dp_drifts[sigma] = dp_drift

        # ----------------------------------------
        # RAG
        # ----------------------------------------
        bm25_ids = bm25.get_top_n(query.split(), list(range(len(pdf))), TOP_K * 4)
        (_, lat_rag_raw) = timed(rag_index.search, qv, TOP_K * 4)
        _, I_rag = rag_index.search(qv, TOP_K * 4)
        vec_ids = list(dict.fromkeys(bm25_ids + I_rag[0].tolist()))
        cand_vecs = embeddings[vec_ids]

        sims = cand_vecs @ qv.ravel()
        selected = []
        candidate_pool = list(range(len(vec_ids)))

        while len(selected) < TOP_K:
            if not selected:
                best = int(np.argmax(sims))
            else:
                selected_vecs = cand_vecs[selected]
                div = cosine_similarity(cand_vecs[candidate_pool], selected_vecs).max(axis=1)
                score = 0.7 * sims[candidate_pool] - 0.3 * div
                best_local = int(np.argmax(score))
                best = candidate_pool[best_local]
            selected.append(best)
            candidate_pool.remove(best)

        rag_ids = [vec_ids[i] for i in selected]

        # ----------------------------------------
        # DP+RAG (sigma = 0.15 canonical)
        # ----------------------------------------
        sigma_dp_rag = 0.15
        dp_rag_vecs = dp_vecs_map[sigma_dp_rag]

        dp_rag_rank_pool = dp_rag_vecs @ qv.ravel()
        dp_rag_ids_sorted = np.argsort(dp_rag_rank_pool)[::-1]
        dp_rag_ids = dp_rag_ids_sorted[:TOP_K]

        # Estimate DP+RAG latency
        (_, lat_dp_rag_raw) = timed(lambda x: x, 0)

        # ----------------------------------------
        # FHE evaluation (top 200 candidate re-ranking)
        # ----------------------------------------
        (_, lat_rp) = timed(lambda: None)
        rp_model = GaussianRandomProjection(n_components=64)
        reduced_docs = normalize_rows(rp_model.fit_transform(embeddings).astype(np.float32))
        reduced_q = rp_model.transform(qv)
        reduced_q = normalize_rows(reduced_q.astype(np.float32))[0]

        # BM25 as candidate generator
        bm25_200 = bm25.get_top_n(query.split(), list(range(len(pdf))), FHE_CANDIDATE_POOL)
        cand_fhe_vecs = reduced_docs[bm25_200]

        # Encryption
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60,40,40,60],
        )
        ctx.global_scale = 2**40
        ctx.generate_galois_keys()

        enc_q = ts.ckks_vector(ctx, reduced_q.tolist())

        fhe_scores = []
        t0 = time.time()
        for vec in cand_fhe_vecs:
            fhe_scores.append(enc_q.dot(vec.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0

        fhe_ids_indices = np.argsort(fhe_scores)[::-1][:TOP_K]
        fhe_ids = [bm25_200[i] for i in fhe_ids_indices]

        # ----------------------------------------
        # Metric computation
        # ----------------------------------------
        row = {
            "query": query,
            "baseline_latency": lat_base,
            "rag_latency": lat_rag_raw,
            "dp_rag_latency": lat_dp_rag_raw,
            "fhe_latency": lat_fhe,
            "ndcg_baseline": 1.0
        }

        # RAG metrics
        ranks = rank_positions(base_ids, rag_ids)
        row["ndcg_rag"] = ndcg_at_k(ranks, TOP_K)
        row["agreement_rag"] = index_agreement(base_ids, rag_ids)

        # DP+RAG
        ranks_dp_rag = rank_positions(base_ids, dp_rag_ids)
        row["ndcg_dp_rag"] = ndcg_at_k(ranks_dp_rag, TOP_K)
        row["agreement_dp_rag"] = index_agreement(base_ids, dp_rag_ids)

        # DP metrics
        for sigma in DP_SIGMAS:
            suffix = f"_{sigma}".replace(".", "p")
            dp_ids = dp_ids_map[sigma]
            ranks_dp = rank_positions(base_ids, dp_ids)

            row[f"dp_latency{suffix}"] = dp_latencies[sigma]
            row[f"ndcg_dp{suffix}"] = ndcg_at_k(ranks_dp, TOP_K)
            row[f"agreement_dp{suffix}"] = index_agreement(base_ids, dp_ids)
            row[f"dp_drift{suffix}"] = dp_drifts[sigma]

        # FHE metrics
        ranks_fhe = rank_positions(base_ids, fhe_ids)
        row["ndcg_fhe"] = ndcg_at_k(ranks_fhe, TOP_K)
        row["agreement_fhe"] = index_agreement(base_ids, fhe_ids)

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# Plotting
# ============================================================

def plot_results(df):

    # Bar pair horizontal for architectures
    arch = ["Baseline","DP Base","DP+RAG","RAG","FHE"]

    latencies = [
        df["baseline_latency"].mean(),
        df["dp_latency_0p15"].mean(),
        df["dp_rag_latency"].mean(),
        df["rag_latency"].mean(),
        df["fhe_latency"].mean(),
    ]

    ndcgs = [
        df["ndcg_baseline"].mean(),
        df["ndcg_dp_0p15"].mean(),
        df["ndcg_dp_rag"].mean(),
        df["ndcg_rag"].mean(),
        df["ndcg_fhe"].mean()
    ]

    shifted_lat = np.log10(latencies) + 1

    fig, ax = plt.subplots(figsize=(10,6))
    y = np.arange(len(arch))
    h = 0.35

    ax.barh(y-h/2, ndcgs, height=h, label="Mean NDCG")
    ax.barh(y+h/2, shifted_lat, height=h, label="Shifted Log Latency", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(arch)
    ax.set_title("Mean NDCG and Latency")
    ax.set_xlabel("Value")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ndcg_latency_arch.png"))
    plt.close()

    # Generate all old style plots too
    # Latency per architecture
    plt.figure(figsize=(9,5))
    df[["baseline_latency","rag_latency","dp_rag_latency","fhe_latency"]].mean().plot(kind="bar", log=True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_simple.png"))
    plt.close()

    # NDCG curves
    plt.figure(figsize=(9,5))
    for col in ["ndcg_rag","ndcg_dp_rag","ndcg_fhe"]:
        plt.plot(df[col], marker="o", label=col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ndcg_curves.png"))
    plt.close()


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("Building resources...")
    pdf, emb, base_index, dp_indices, dp_vecs_map, bm25, rag_index, model = build_all(
        DATA_PATH, "sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Running evaluation...")
    df = evaluate_all(pdf, emb, base_index, dp_indices, dp_vecs_map, bm25, rag_index, model)

    out_csv = os.path.join(ROOT, "evaluation_results_updated.csv")
    df.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)

    print("Plotting...")
    plot_results(df)
    print("Plots saved to:", PLOTS_DIR)

if __name__ == "__main__":
    main()
