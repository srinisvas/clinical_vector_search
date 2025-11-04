import argparse
import sys

# sys.path.append(".")
from pipeline.pipeline_mode import mode_baseline, mode_dp, mode_fhe, mode_rag

def parse_args():
    p = argparse.ArgumentParser(description="Privacy-Preserving Vector-Based Semantic Search for Clinical Text")
    p.add_argument("--mode", required=True, choices=["baseline", "dp", "fhe", "rag"],
                   help="Which track to run.")
    p.add_argument("--data_path", required=True, help="Path to MTSamples CSV (medical_transcriptions.csv).")
    p.add_argument("--index_path", default="./faiss_mtsamples.faiss", help="Where to save FAISS index.")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Sentence-Transformers model (try: 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb').")
    p.add_argument("--topk", type=int, default=10, help="Top-k results to return.")
    p.add_argument("--query", type=str, default="", help="Optional test query. use only after index is built.")
    # DP
    p.add_argument("--sigma", type=float, default=0.15, help="Gaussian noise stddev for DP mode.")
    # FHE
    p.add_argument("--subset", type=int, default=300, help="FHE POC subset size (tiny).")
    # RAG tuning
    p.add_argument("--hnsw_M", type=int, default=32, help="HNSW M (graph degree).")
    p.add_argument("--efC", type=int, default=200, help="HNSW efConstruction.")
    p.add_argument("--efS", type=int, default=128, help="HNSW efSearch (query-time).")
    p.add_argument("--candidate_k", type=int, default=128, help="Vector candidate pool before re-rank.")
    p.add_argument("--enable_hybrid", action="store_true", help="Enable BM25+vector hybrid.")
    p.add_argument("--bm25_topk", type=int, default=120, help="BM25 candidate pool size.")
    p.add_argument("--mmr_lambda", type=float, default=0.5, help="MMR trade-off between relevance and diversity.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "baseline":
        mode_baseline(args)
    elif args.mode == "dp":
        mode_dp(args)
    elif args.mode == "fhe":
        mode_fhe(args)
    elif args.mode == "rag":
        mode_rag(args)