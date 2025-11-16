import argparse
from pipeline.pipeline_mode import mode_baseline, mode_dp, mode_fhe, mode_rag


def parse_args():
    p = argparse.ArgumentParser(description="Privacy-Preserving Clinical Semantic Search")

    p.add_argument("--mode", required=True, choices=["baseline", "dp", "fhe", "rag"])
    p.add_argument("--data_path", required=True)
    p.add_argument("--index_path", default="./faiss_mtsamples.faiss")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--query", type=str, default="")

    # DP
    p.add_argument("--sigma", type=float, default=0.15)

    # FHE
    p.add_argument("--subset", type=int, default=300)

    # RAG
    p.add_argument("--hnsw_M", type=int, default=32)
    p.add_argument("--efC", type=int, default=200)
    p.add_argument("--efS", type=int, default=128)
    p.add_argument("--candidate_k", type=int, default=128)
    p.add_argument("--enable_hybrid", action="store_true")
    p.add_argument("--bm25_topk", type=int, default=120)
    p.add_argument("--mmr_lambda", type=float, default=0.5)

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
