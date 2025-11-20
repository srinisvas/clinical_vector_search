import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NOTE: keep imports minimal in evaluation to reduce native lib pressure on macOS
from sentence_transformers import SentenceTransformer
import faiss
import tenseal as ts
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# Make sure `src` (parent of `pipeline`) is on sys.path when run as module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)  # .../src
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from pipeline.utils import normalize_rows, norm_vec

# Immediately make FAISS single-threaded to avoid macOS segfaults
try:
    faiss.omp_set_num_threads(1)
except Exception:
    pass

# Also limit sentence-transformers parallelism via environment
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "dataset", "medical_transcriptions.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
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

TOP_K = 10

SAFE_MODE = False  # set True if native segfaults persist; modes will degrade gracefully

# Define the DP noise sigmas we want to sweep for pure DP mode
DP_SIGMAS = [0.05, 0.1, 0.15, 0.2]


# ============================================================
# Utility Functions
# ============================================================

def timed(fn, *args, **kwargs):
    """Time a function call and return (result, latency_ms)."""
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, (time.time() - t0) * 1000.0


def ndcg_at_k(ranks, k):
    """Compute NDCG@k assuming rank positions as relevance."""
    ranks = np.array(ranks)[:k]
    gains = 1.0 / np.log2(np.arange(2, len(ranks) + 2))
    return np.sum(gains * ranks) / np.sum(gains)


def index_agreement(ref, other):
    """Percentage of exact index matches."""
    ref = list(ref)
    other = list(other)
    matches = sum(1 for a, b in zip(ref, other) if a == b)
    return matches / len(ref)


# ============================================================
# Build all embeddings + indexes fresh
# ============================================================

def build_all(data_path, model_name):
    print("Loading dataset with pandas (no Spark)...")
    raw = pd.read_csv(data_path)

    # Normalize column names to lowercase for robustness
    raw.columns = [c.lower() for c in raw.columns]

    # Map original MTSamples schema to the fields we need
    # Header: Name,Gender,Age,City,description,medical_specialty,sample_name,transcription,keywords,Age_extracted
    col_name = "name" if "name" in raw.columns else None
    col_gender = "gender" if "gender" in raw.columns else None
    col_age = "age" if "age" in raw.columns else None
    col_city = "city" if "city" in raw.columns else None
    col_specialty = "medical_specialty" if "medical_specialty" in raw.columns else None
    col_trans = "transcription" if "transcription" in raw.columns else None

    missing_core = [c for c, v in {
        "name": col_name,
        "gender": col_gender,
        "age": col_age,
        "city": col_city,
        "medical_specialty": col_specialty,
        "transcription": col_trans,
    }.items() if v is None]
    if missing_core:
        raise ValueError(f"CSV missing expected MTSamples columns: {missing_core}")

    # Build a working DataFrame with unified column names
    pdf = pd.DataFrame({
        "name": raw[col_name],
        "gender": raw[col_gender],
        "age": raw[col_age],
        "city": raw[col_city],
        "medical_specialty": raw[col_specialty],
        "transcription": raw[col_trans],
    })

    # Construct `text` exactly like load_mtsamples_df: specialty + transcription
    pdf["text"] = pdf.apply(
        lambda x: f"{x['medical_specialty']}, {x['transcription']}"
        if pd.notnull(x["medical_specialty"])
        else x["transcription"],
        axis=1,
    )

    # Drop duplicates on text to match pipeline behavior
    pdf = pdf.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("Embedding dataset...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        pdf["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )
    embeddings = normalize_rows(np.array(embeddings, dtype=np.float32))

    pdf["vec"] = list(embeddings)

    # ------------ Baseline index (FlatIP) -------------
    d = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(d)
    base_index.add(embeddings)

    # ------------ DP Indexes with ALL attributes and multiple sigmas -------------
    # Use all columns except `text` and `vec` as attributes
    attr_cols = [c for c in pdf.columns if c not in ["text", "vec"]]

    # Join all attribute fields into one string per row
    attr_texts = pdf[attr_cols].astype(str).apply(lambda row: " ".join(row.values), axis=1).tolist()

    attr_emb = model.encode(attr_texts, batch_size=32, show_progress_bar=False)
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

    # For DP+RAG we keep using sigma=0.15 specifically
    dp_rag_sigma = 0.15
    dp_rag_index = dp_indices[dp_rag_sigma]

    # ------------ RAG Structures (BM25 + FAISS + MMR) -------------
    tokenized = [t.lower().split() for t in pdf["text"]]
    bm25 = BM25Okapi(tokenized)

    # Use FlatIP instead of HNSW for macOS stability
    rag_index = faiss.IndexFlatIP(d)
    rag_index.add(embeddings)

    return pdf, embeddings, base_index, dp_indices, dp_rag_index, bm25, rag_index, model


# ============================================================
# Compute metrics
# ============================================================

def evaluate_all(pdf, embeddings, base_index, dp_indices, dp_rag_index, bm25, rag_index, model):
    results = []

    for query in QUERIES:
        print(f"\nEvaluating: {query}")

        # ------------------------------------
        # Query embedding
        # ------------------------------------
        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # ------------------------------------
        # BASELINE
        # ------------------------------------
        if SAFE_MODE:
            _, lat_base = (None, 0.0)
            I_base = np.arange(min(TOP_K, embeddings.shape[0])).reshape(1, -1)
        else:
            (_, lat_base) = timed(base_index.search, qv, TOP_K)
            _, I_base = base_index.search(qv, TOP_K)
        base_ids = I_base[0]

        # ------------------------------------
        # DP modes sweep
        # ------------------------------------
        dp_latencies = {}
        dp_id_lists = {}
        dp_drifts = {}

        text_dim = qv.shape[1]

        for sigma, dp_index in dp_indices.items():
            dp_dim = dp_index.d
            if dp_dim > text_dim:
                attr_dim = dp_dim - text_dim
                qv_zero_attr = np.zeros((1, attr_dim), dtype=np.float32)
                qv_dp = normalize_rows(np.hstack([qv * 0.7, qv_zero_attr * 0.3]))
            else:
                qv_dp = qv

            if SAFE_MODE:
                _, lat_dp = (None, 0.0)
                I_dp = I_base.copy()
            else:
                (_, lat_dp) = timed(dp_index.search, qv_dp, TOP_K)
                _, I_dp = dp_index.search(qv_dp, TOP_K)

            dp_ids = I_dp[0]
            # cosine_similarity requires matching feature dimensions; guard against mismatch
            if qv.shape[1] == qv_dp.shape[1]:
                dp_drift = float(cosine_similarity(qv, qv_dp)[0][0])
            else:
                dp_drift = 1.0  # treat as no drift when we had to pad/reshape

            dp_latencies[sigma] = lat_dp
            dp_id_lists[sigma] = dp_ids
            dp_drifts[sigma] = dp_drift

        # ------------------------------------
        # FHE MODE (minimal synthetic demo)
        # ------------------------------------
        if SAFE_MODE:
            lat_fhe = 0.0
            fhe_ids = []
        else:
            d_target = 64
            R = np.random.normal(0, 1 / np.sqrt(qv.shape[1]), size=(qv.shape[1], d_target)).astype(np.float32)

            qv_small = qv @ R
            qv_small = normalize_rows(qv_small)[0]
            qv_small = norm_vec(qv_small.astype(np.float32))

            # Only a handful of synthetic vectors to keep TenSEAL work tiny
            small_vecs = normalize_rows(np.random.randn(10, d_target).astype(np.float32))

            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60],
            )
            ctx.global_scale = 2**40
            ctx.generate_galois_keys()

            enc_q = ts.ckks_vector(ctx, qv_small.tolist())

            fhe_scores = []
            t0 = time.time()
            for v in small_vecs:
                fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
            lat_fhe = (time.time() - t0) * 1000.0

            fhe_ids = np.argsort(fhe_scores)[::-1][:TOP_K]

        # ------------------------------------
        # RAG MODE (BM25 + FlatIP + MMR)
        # ------------------------------------
        if SAFE_MODE:
            bm25_ids = list(range(min(TOP_K * 4, embeddings.shape[0])))
            lat_rag_raw = 0.0
            vec_ids = bm25_ids
        else:
            bm25_ids = bm25.get_top_n(query.split(), list(range(len(pdf))), TOP_K * 4)

            (_, lat_rag_raw) = timed(rag_index.search, qv, TOP_K * 4)
            _, I_rag = rag_index.search(qv, TOP_K * 4)
            vec_ids = I_rag[0].tolist()

        # merge candidates
        cand = list(dict.fromkeys(bm25_ids + vec_ids))
        cand_vecs = embeddings[cand]

        q = qv.ravel()
        sims = cand_vecs @ q

        # MMR
        selected = []
        candidate_ids = list(range(len(cand)))

        while len(selected) < TOP_K:
            if len(selected) == 0:
                best = int(np.argmax(sims))
            else:
                selected_vecs = cand_vecs[selected]
                diversity = cosine_similarity(cand_vecs[candidate_ids], selected_vecs).max(axis=1)
                score = 0.7 * sims[candidate_ids] - 0.3 * diversity
                best_local = int(np.argmax(score))
                best = candidate_ids[best_local]

            selected.append(best)
            candidate_ids.remove(best)

        rag_ids = [cand[i] for i in selected]

        # ------------------------------------
        # DP + Fastened RAG MODE
        # ------------------------------------
        # 1) Candidate generation with BM25 (same as RAG)
        if SAFE_MODE:
            bm25_ids_dp_rag = list(range(min(TOP_K * 4, embeddings.shape[0])))
            lat_dp_rag_raw = 0.0
            vec_ids_dp_rag = bm25_ids_dp_rag
        else:
            bm25_ids_dp_rag = bm25.get_top_n(query.split(), list(range(len(pdf))), TOP_K * 4)

            (_, lat_dp_rag_raw) = timed(rag_index.search, qv_dp[:, :rag_index.d], TOP_K * 4)
            _, I_dp_rag = rag_index.search(qv_dp[:, :rag_index.d], TOP_K * 4)
            vec_ids_dp_rag = I_dp_rag[0].tolist()

        cand_dp_rag = list(dict.fromkeys(bm25_ids_dp_rag + vec_ids_dp_rag))
        cand_vecs_dp_rag = embeddings[cand_dp_rag]

        q = qv.ravel()
        sims_dp_rag = cand_vecs_dp_rag @ q

        selected_dp_rag = []
        candidate_ids_dp_rag = list(range(len(cand_dp_rag)))

        while len(selected_dp_rag) < TOP_K and candidate_ids_dp_rag:
            if len(selected_dp_rag) == 0:
                best = int(np.argmax(sims_dp_rag))
            else:
                selected_vecs_dp_rag = cand_vecs_dp_rag[selected_dp_rag]
                diversity_dp_rag = cosine_similarity(
                    cand_vecs_dp_rag[candidate_ids_dp_rag],
                    selected_vecs_dp_rag,
                ).max(axis=1)
                score_dp_rag = 0.7 * sims_dp_rag[candidate_ids_dp_rag] - 0.3 * diversity_dp_rag
                best_local = int(np.argmax(score_dp_rag))
                best = candidate_ids_dp_rag[best_local]

            selected_dp_rag.append(best)
            candidate_ids_dp_rag.remove(best)

        dp_rag_ids = [cand_dp_rag[i] for i in selected_dp_rag]

        # ------------------------------------
        # Metrics
        # ------------------------------------
        # rankings for NDCG
        def rank_positions(ref, pred):
            mapping = {doc_id: i + 1 for i, doc_id in enumerate(ref)}
            return [mapping.get(x, 0) for x in pred]

        # Compute metrics for each DP sigma
        row = {
            "query": query,
            "baseline_latency": lat_base,
            "rag_latency": lat_rag_raw,
            "dp_rag_latency": lat_dp_rag_raw,
        }

        # RAG-only metrics (baseline vs rag)
        ranks_rag = rank_positions(base_ids, rag_ids)
        ndcg_rag = ndcg_at_k(ranks_rag, TOP_K)
        agreement_rag = index_agreement(base_ids, rag_ids)
        row["ndcg_rag"] = ndcg_rag
        row["agreement_rag"] = agreement_rag

        # DP+RAG metrics
        ranks_dp_rag = rank_positions(base_ids, dp_rag_ids)
        ndcg_dp_rag = ndcg_at_k(ranks_dp_rag, TOP_K)
        agreement_dp_rag = index_agreement(base_ids, dp_rag_ids)
        row["ndcg_dp_rag"] = ndcg_dp_rag
        row["agreement_dp_rag"] = agreement_dp_rag

        # Per-sigma DP metrics
        for sigma in DP_SIGMAS:
            dp_ids = dp_id_lists[sigma]
            dp_latency = dp_latencies[sigma]
            dp_drift = dp_drifts[sigma]

            ranks_dp = rank_positions(base_ids, dp_ids)
            ndcg_dp = ndcg_at_k(ranks_dp, TOP_K)
            agreement_dp = index_agreement(base_ids, dp_ids)

            # encode fields with sigma in the key
            suffix = f"_{sigma}".replace(".", "p")
            row[f"dp_latency{suffix}"] = dp_latency
            row[f"ndcg_dp{suffix}"] = ndcg_dp
            row[f"agreement_dp{suffix}"] = agreement_dp
            row[f"dp_drift{suffix}"] = dp_drift

        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# Plotting
# ============================================================

def plot_results(df):
    """Plot latency and quality metrics.

    Note: After introducing DP sweeps, per-sigma DP metrics are encoded
    as dp_latency_<sigma>, ndcg_dp_<sigma>, etc. We visualize:
    - Latency: baseline vs RAG vs DP+RAG
    - NDCG & agreement: RAG and DP+RAG
    - DP sweep: NDCG across all four DP sigmas.
    """
    # LATENCY (log scale) – baseline, RAG, DP+RAG
    plt.figure(figsize=(9, 5))
    latency_cols = [
        "baseline_latency",
        "rag_latency",
        "dp_rag_latency",
    ]
    # Guard against missing columns in case of partial runs
    latency_cols = [c for c in latency_cols if c in df.columns]
    df[latency_cols].mean().plot(kind="bar", log=True)
    plt.title("Latency (log scale)")
    plt.ylabel("Latency (ms, log)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_log.png"))
    plt.close()

    # NDCG – RAG and DP+RAG
    plt.figure(figsize=(9, 5))
    if "ndcg_rag" in df.columns:
        plt.plot(df["ndcg_rag"], marker="o", label="RAG")
    if "ndcg_dp_rag" in df.columns:
        plt.plot(df["ndcg_dp_rag"], marker="o", label="DP+RAG")
    plt.title("NDCG Comparison (RAG vs DP+RAG)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ndcg.png"))
    plt.close()

    # Agreement – RAG and DP+RAG
    plt.figure(figsize=(9, 5))
    if "agreement_rag" in df.columns:
        plt.plot(df["agreement_rag"], marker="o", label="RAG")
    if "agreement_dp_rag" in df.columns:
        plt.plot(df["agreement_dp_rag"], marker="o", label="DP+RAG")
    plt.title("Index Agreement with Baseline (RAG vs DP+RAG)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "agreement.png"))
    plt.close()

    # DP sweep NDCG for each sigma
    plt.figure(figsize=(9, 5))
    for sigma in DP_SIGMAS:
        suffix = f"_{sigma}".replace(".", "p")
        col = f"ndcg_dp{suffix}"
        if col in df.columns:
            plt.plot(df[col], marker="o", label=f"DP σ={sigma}")
    plt.title("DP NDCG Sweep Across Sigmas")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ndcg_dp_sweep.png"))
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("Building full environment fresh...")

    pdf, embeddings, base_index, dp_indices, dp_rag_index, bm25, rag_index, model = build_all(
        DATA_PATH,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    df = evaluate_all(
        pdf,
        embeddings,
        base_index,
        dp_indices,
        dp_rag_index,
        bm25,
        rag_index,
        model
    )

    out = os.path.join(PROJECT_ROOT, "evaluation_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved metrics to {out}")

    plot_results(df)
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
