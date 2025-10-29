from sentence_transformers import SentenceTransformer

def build_spark(app_name="ClinicalVectorSearch"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    return spark

def clean_text(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.replace("\r", " ").replace("\n", " ")
    x = " ".join(x.split())
    return x

def embed_partition(texts: List[str], model_name: str) -> List[List[float]]:
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=16, show_progress_bar=False)
    vecs = normalize_rows(np.array(vecs, dtype=np.float32))
    return vecs.tolist()

def add_noise_vec(vec: List[float], sigma: float) -> List[float]:
    v = np.array(vec, dtype=np.float32)
    v = v + np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)
    v = norm_vec(v)
    return v.tolist()