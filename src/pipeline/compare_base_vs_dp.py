import numpy as np
import pandas as pd
import faiss
from time import time
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Utility Functions

def normalize_rows(x):
    return normalize(x, norm="l2")

def overlap(a, b):
    return len(set(a).intersection(set(b))) / len(a)

def print_results(title, pdf, D, I, tat, k=5):
    print(f"\n--*-- {title} (TAT: {round(tat, 2)} ms)--*--")
    for rank, idx in enumerate(I[0][:k]):
        snippet = pdf.iloc[idx]["text"][:150].replace("\n", " ")
        name, gender, age, city = pdf.iloc[idx][["name","gender","age","city"]]
        print(f"{rank+1:>2}. score={D[0][rank]:.4f} | {name}, {gender}, {age}, {city} :: {snippet}...")

#Load FAISS Indexes
print("Loading FAISS indexes...")

index_base = faiss.read_index("resources/faiss_mtsamples.faiss")
index_dp = faiss.read_index("resources/faiss_mtsamples_dp_sigma015_1.faiss")

print("Loaded baseline and DP indexes successfully.")

# Load or Reuse Metadata 
pdf = pd.read_parquet("resources/mt_samples_embeddings.parquet")

# Model for Queries
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Demo Queries for Comparison
queries = [
    "Reports mentioning diabetes and hypertension",
    "Female patients greater than 60 from Phoenix",
    "Male patients treated in Chicago hospitals",
    "Elderly patient with chest pain",
    "Young adults admitted to San Francisco hospitals"
]

# main run

if __name__ == "__main__":

    for q in queries:
        print("\n" + "-"*80)
        print(f"Query: {q}")

        # Text-only embedding (baseline)
        qv_text = model.encode([q])
        qv_text = normalize_rows(qv_text.astype(np.float32))

        # Attribute query embedding (for DP combined search)
        qv_attr = model.encode([q])
        qv_attr = normalize_rows(qv_attr.astype(np.float32))

        # Combine text + attribute vectors
        qv_combined = np.hstack([qv_text * 0.7, qv_attr * 0.3])
        qv_combined = normalize_rows(qv_combined)

        # Perform base rag search with latency measure
        base_start_time = time()
        D_base, I_base = index_base.search(qv_text, k=5)
        base_end_time = time()
        base_tat = (base_end_time - base_start_time) * 1000.0
        
        # Perform DP rag search with latency measure
        dp_start_time = time()
        D_dp, I_dp = index_dp.search(qv_combined, k=5)
        dp_end_time = time()
        dp_tat = (dp_end_time - dp_start_time) * 1000.0

        # Print results
        print_results("Baseline RAG", pdf, D_base, I_base, base_tat)
        print_results("DP-Attribute RAG", pdf, D_dp, I_dp, dp_tat)

        print(f"\nOverlap (Top-5): {overlap(I_base[0], I_dp[0]):.2f}")