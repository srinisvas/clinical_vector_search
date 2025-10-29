import faiss
from rank_bm25 import BM25Okapi

def load_mtsamples_df(spark, data_path: str) -> pd.DataFrame:
    # MTSamples common columns: "medical_specialty", "transcription", (sometimes "description")
    sdf = spark.read.csv(data_path, header=True, multiLine=True, escape='"')
    # normalize column names we care about
    cols = [c.lower() for c in sdf.columns]
    mapping = {}
    if "medical_specialty" in cols:
        mapping[sdf.columns[cols.index("medical_specialty")]] = "medical_specialty"
    if "transcription" in cols:
        mapping[sdf.columns[cols.index("transcription")]] = "text"
    elif "sample" in cols:
        mapping[sdf.columns[cols.index("sample")]] = "text"
    else:
        # fallback: use first textual column
        mapping[sdf.columns[0]] = "text"

    sdf = sdf.select([F.col(k).alias(v) for k, v in mapping.items()])
    sdf = sdf.withColumn("text", F.col("text").cast(T.StringType()))
    sdf = sdf.withColumn("text", F.udf(clean_text, T.StringType())("text"))
    sdf = sdf.na.drop(subset=["text"])
    pdf = sdf.toPandas()
    pdf = pdf.drop_duplicates(subset=["text"]).reset_index(drop=True)
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


def bm25_topk(corpus_texts: List[str], query: str, topk: int = 100) -> List[int]:
    if not HAS_BM25:
        return list(range(0, min(topk, len(corpus_texts))))
    tokenized = [t.lower().split() for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    idx = np.argsort(scores)[::-1][:topk].tolist()
    return idx