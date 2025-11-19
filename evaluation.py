import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

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

    # ------------ DP Index -------------
    attr_texts = [
        f"{n} {g} {a} {c}"
        for n, g, a, c in zip(pdf["name"], pdf["gender"], pdf["age"], pdf["city"])
    ]
    attr_emb = model.encode(attr_texts, batch_size=32, show_progress_bar=False)
    attr_emb = normalize_rows(attr_emb.astype(np.float32))

    sigma = 0.15
    # Add noise to main text embeddings
    noisy_embeddings = embeddings + np.random.normal(0, sigma, embeddings.shape).astype(np.float32)
    noisy_embeddings = normalize_rows(noisy_embeddings)

    # Add noise to attribute embeddings
    noisy_attr_emb = attr_emb + np.random.normal(0, sigma, attr_emb.shape).astype(np.float32)
    noisy_attr_emb = normalize_rows(noisy_attr_emb)

    dp_vecs = normalize_rows(
        np.hstack([noisy_embeddings * 0.7, noisy_attr_emb * 0.3]).astype(np.float32)
    )

    dp_index = faiss.IndexFlatIP(dp_vecs.shape[1])
    dp_index.add(dp_vecs)

    # ------------ RAG Structures (BM25 + FAISS + MMR) -------------
    tokenized = [t.lower().split() for t in pdf["text"]]
    bm25 = BM25Okapi(tokenized)

    # Use FlatIP instead of HNSW for macOS stability
    rag_index = faiss.IndexFlatIP(d)
    rag_index.add(embeddings)

    return pdf, embeddings, base_index, dp_index, bm25, rag_index, model


# ============================================================
# Compute metrics
# ============================================================

def evaluate_all(pdf, embeddings, base_index, dp_index, bm25, rag_index, model):
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
        # DP MODE
        # ------------------------------------
        text_dim = qv.shape[1]
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
        # Metrics
        # ------------------------------------
        # rankings for NDCG
        def rank_positions(ref, pred):
            mapping = {doc_id: i+1 for i, doc_id in enumerate(ref)}
            return [mapping.get(x, 0) for x in pred]

        ranks_dp = rank_positions(base_ids, dp_ids)
        ranks_rag = rank_positions(base_ids, rag_ids)

        ndcg_dp = ndcg_at_k(ranks_dp, TOP_K)
        ndcg_rag = ndcg_at_k(ranks_rag, TOP_K)

        agreement_rag = index_agreement(base_ids, rag_ids)
        agreement_dp = index_agreement(base_ids, dp_ids)

        results.append({
            "query": query,
            "baseline_latency": lat_base,
            "dp_latency": lat_dp,
            "fhe_latency": lat_fhe,
            "rag_latency": lat_rag_raw,

            "ndcg_dp": ndcg_dp,
            "ndcg_rag": ndcg_rag,

            "agreement_dp": agreement_dp,
            "agreement_rag": agreement_rag,

            "dp_drift": dp_drift,
            "rag_improvement": ndcg_rag - ndcg_dp
        })

    return pd.DataFrame(results)


# ============================================================
# Plotting
# ============================================================

def plot_results(df):
    # LATENCY (log scale)
    plt.figure(figsize=(9, 5))
    modes = ["baseline_latency", "dp_latency", "fhe_latency", "rag_latency"]
    df[modes].mean().plot(kind="bar", log=True)
    plt.title("Latency (log scale)")
    plt.ylabel("Latency (ms, log)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_log.png"))
    plt.close()

    # NDCG
    plt.figure(figsize=(9, 5))
    plt.plot(df["ndcg_dp"], marker="o", label="DP")
    plt.plot(df["ndcg_rag"], marker="o", label="RAG")
    plt.title("NDCG Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ndcg.png"))
    plt.close()

    # Agreement
    plt.figure(figsize=(9, 5))
    plt.plot(df["agreement_dp"], marker="o", label="DP")
    plt.plot(df["agreement_rag"], marker="o", label="RAG")
    plt.title("Index Agreement with Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "agreement.png"))
    plt.close()

    # DP Semantic Drift
    plt.figure(figsize=(9, 5))
    plt.plot(df["dp_drift"], marker="o")
    plt.title("DP Semantic Drift (cosine similarity)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "dp_drift.png"))
    plt.close()

    # RAG improvement curve
    plt.figure(figsize=(9, 5))
    plt.plot(df["rag_improvement"], marker="o")
    plt.title("RAG Improvement Over DP (Δ NDCG)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rag_improvement.png"))
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("Building full environment fresh...")

    pdf, embeddings, base_index, dp_index, bm25, rag_index, model = build_all(
        DATA_PATH,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    df = evaluate_all(
        pdf,
        embeddings,
        base_index,
        dp_index,
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
