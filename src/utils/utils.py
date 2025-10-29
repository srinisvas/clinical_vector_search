import time

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity


def norm_vec(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)

def normalize_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / n

def mmr_rerank(query_vec: np.ndarray, cand_vecs: np.ndarray, cand_ids: List[int],
               k: int = 10, lambda_param: float = 0.5) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) re-ranking.
    """
    chosen = []
    candidates = list(range(len(cand_ids)))
    sim_to_query = (cand_vecs @ query_vec.reshape(-1, 1)).ravel()

    while len(chosen) < min(k, len(cand_ids)):
        if not chosen:
            idx = int(np.argmax(sim_to_query))
            chosen.append(idx)
            candidates.remove(idx)
            continue
        # diversity term: max sim to selected
        selected_vecs = cand_vecs[chosen]
        sim_to_selected = cosine_similarity(cand_vecs[candidates], selected_vecs).max(axis=1)
        score = lambda_param * sim_to_query[candidates] - (1 - lambda_param) * sim_to_selected
        pick_local = int(np.argmax(score))
        idx = candidates[pick_local]
        chosen.append(idx)
        candidates.remove(idx)

    return [cand_ids[i] for i in chosen]


def timer():
    start = time.time()
    def end(msg: str = ""):
        dt = (time.time() - start) * 1000.0
        print(f"[TIMER] {msg} {dt:.1f} ms")
    return end
