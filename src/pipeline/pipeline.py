import pandas as pd
import numpy as np
from pyspark.sql import functions as F, types as T
from time import time
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from typing import List, Tuple

from pipeline.utils import normalize_rows, clean_text


# -------------------------------------------------------
# Load & Clean Dataset with Spark
# -------------------------------------------------------

def load_mtsamples_df(spark, data_path: str) -> pd.DataFrame:
    """
    Loads MTSamples CSV using Spark, cleans text, merges medical_specialty
    into the text field, removes duplicates, and returns a pandas DataFrame.
    Fully mac-compatible.
    """

    spark_start = time()

    sdf = (
        spark.read
        .csv(data_path, header=True, multiLine=True, escape='"')
    )
    print(f"Rows Loaded: {sdf.count()}")

    # Lowercase column names
    cols = [c.lower() for c in sdf.columns]

    # Required MTSamples fields
    mapping = {
        sdf.columns[cols.index("name")]: "name",
        sdf.columns[cols.index("gender")]: "gender",
        sdf.columns[cols.index("age")]: "age",
        sdf.columns[cols.index("city")]: "city",
        sdf.columns[cols.index("medical_specialty")]: "medical_specialty",
        sdf.columns[cols.index("transcription")]: "text",
    }

    # Select and rename
    sdf = sdf.select([F.col(k).alias(v) for k, v in mapping.items()])

    # Clean text
    sdf = sdf.withColumn("text", F.col("text").cast(T.StringType()))
    sdf = sdf.withColumn("text", F.udf(clean_text, T.StringType())("text"))

    # Drop empty rows
    sdf = sdf.na.drop(subset=["text"])

    # Convert to Pandas
    pdf = sdf.toPandas()

    # Merge specialty with transcription
    pdf["text"] = pdf.apply(
        lambda x: f"{x['medical_specialty']}, {x['text']}"
        if pd.notnull(x["medical_specialty"])
        else x["text"],
        axis=1,
    )

    # Remove duplicates
    pdf = pdf.drop_duplicates(subset=["text"]).reset_index(drop=True)

    spark_tat = round((time() - spark_start) * 1000, 2)
    print(f"Rows after preprocessing: {pdf.shape}")
    print(f"Spark load time: {spark_tat} ms")

    return pdf


# -------------------------------------------------------
# Embedding Builder
# -------------------------------------------------------

def build_embeddings_with_spark(
    pdf: pd.DataFrame,
    model_name: str,
    save_path: str = None,
    batch_col: str = "text"
) -> pd.DataFrame:
    """
    Builds sentence embeddings on macOS using local CPU.
    """

    model = SentenceTransformer(model_name)
    vecs = model.encode(
        pdf[batch_col].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )

    vecs = normalize_rows(np.array(vecs, dtype=np.float32))
    pdf["vec"] = list(vecs)

    if save_path:
        print(f"Saving embeddings to {save_path}")
        pdf.to_parquet(save_path, index=False)

    return pdf


# -------------------------------------------------------
# FAISS Index Builder
# -------------------------------------------------------

def build_faiss_index(
    vectors: np.ndarray,
    index_path: str,
    hnsw: bool = False,
    hnsw_M: int = 32,
    efC: int = 200
):
    """
    Builds FAISS FlatIP or HNSW index.
    Designed for macOS FAISS CPU.
    """

    vectors = vectors.astype(np.float32)
    d = vectors.shape[1]

    if hnsw:
        index = faiss.IndexHNSWFlat(d, hnsw_M)
        index.hnsw.efConstruction = efC
        # efSearch is applied at query time
    else:
        index = faiss.IndexFlatIP(d)

    index.add(vectors)
    faiss.write_index(index, index_path)

    return index


# -------------------------------------------------------
# FAISS Search
# -------------------------------------------------------

def search_faiss(
    index,
    query_vec: np.ndarray,
    k: int = 10,
    efSearch: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs FAISS search.
    For HNSW, efSearch is applied dynamically at query time.
    """

    if efSearch is not None and hasattr(index, "hnsw"):
        index.hnsw.efSearch = efSearch

    query_vec = query_vec.astype(np.float32)
    D, I = index.search(query_vec, k)
    return D, I


# -------------------------------------------------------
# BM25 Retrieval
# -------------------------------------------------------

def bm25_topk(corpus_texts: List[str], query: str, topk: int = 100) -> List[int]:
    """
    Runs BM25 lexical search. Tokenizes by splitting on whitespace.
    """

    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(scores)[::-1][:topk]

    return idx.tolist()