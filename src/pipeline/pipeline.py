import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, functions as F, types as T
from sentence_transformers import SentenceTransformer
import faiss
# from rank_bm25 import BM25Okapi
from typing import List, Tuple
import os

from pipeline.utils import normalize_rows, clean_text

def load_mtsamples_df(spark, data_path: str) -> pd.DataFrame:

    # import pdb; pdb.set_trace()
    sdf = spark.read.csv(data_path, header=True, multiLine=True, escape='"')
    print(f"Rows Loaded: {sdf.count()}")
    # normalize column names
    cols = [c.lower() for c in sdf.columns]
    print("sdf cols", cols)

    # create mapping from existing columns to standard names
    mapping = {}
    mapping[sdf.columns[cols.index("name")]] = "name"
    mapping[sdf.columns[cols.index("gender")]] = "gender"
    mapping[sdf.columns[cols.index("age")]] = "age"
    mapping[sdf.columns[cols.index("city")]] = "city"
    mapping[sdf.columns[cols.index("medical_specialty")]] = "medical_specialty"
    mapping[sdf.columns[cols.index("transcription")]] = "text"

    # select and rename
    sdf = sdf.select([F.col(k).alias(v) for k, v in mapping.items()])

    # cast text to string and clean
    sdf = sdf.withColumn("text", F.col("text").cast(T.StringType()))
    sdf = sdf.withColumn("text", F.udf(clean_text, T.StringType())("text"))

    # drop rows with null text
    sdf = sdf.na.drop(subset=["text"])
    pdf = sdf.toPandas()

    #combine medical specialty with text (makes unique)
    pdf["text"] = pdf.apply(lambda x: f"{x['medical_specialty']}, {x['text']}" if pd.notnull(x['medical_specialty']) else x['text'], axis=1)

    # pdf.to_csv("debug_mtsamples_cleaned.csv", index=False)
    pdf = pdf.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Rows after preprocessing: {pdf.shape}")

    return pdf


def build_embeddings_with_spark(pdf: pd.DataFrame, model_name: str, batch_col: str = "text") -> pd.DataFrame:
    # Use local (driver) batching to avoid serializing models to workers (simpler & reliable for 5k–50k docs)
    # If you need true distributed inference, convert to RDD partitions and call embed_partition.
    model = SentenceTransformer(model_name)
    vecs = model.encode(pdf[batch_col].tolist(), batch_size=32, show_progress_bar=True)
    vecs = normalize_rows(np.array(vecs, dtype=np.float32))
    pdf["vec"] = list(vecs)
    return pdf


def build_faiss_index(vectors: np.ndarray, index_path: str, hnsw: bool = False, hnsw_M: int = 32, efC: int = 200):
    d = vectors.shape[1]
    if hnsw:
        index = faiss.IndexHNSWFlat(d, hnsw_M)  # cosine via normalized vectors + inner product
        index.hnsw.efConstruction = efC
    else:
        index = faiss.IndexFlatIP(d)  # inner product (cosine if normalized)
    index.add(vectors)
    faiss.write_index(index, index_path)
    return index


def search_faiss(index, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    D, I = index.search(query_vec.astype(np.float32), k)
    return D, I


# def bm25_topk(corpus_texts: List[str], query: str, topk: int = 100) -> List[int]:
#     if not HAS_BM25:
#         return list(range(0, min(topk, len(corpus_texts))))
#     tokenized = [t.lower().split() for t in corpus_texts]
#     bm25 = BM25Okapi(tokenized)
#     scores = bm25.get_scores(query.lower().split())
#     idx = np.argsort(scores)[::-1][:topk].tolist()
#     return idx