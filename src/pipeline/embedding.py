from pyspark.sql import SparkSession, functions as F, types as T
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np

from pipeline.utils import normalize_rows

def build_spark(app_name="ClinicalVectorSearch"):

    host = "spark://spark:7077"
    print(f"Building Spark Session..., host: {host}")
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(host)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.executorEnv.PYTHONPATH", "/app/src")
        # .config("spark.driver.host", "host.docker.internal")
        # .config("spark.driver.port", "4041")
        # .config("spark.driver.bindAddress", "0.0.0.0")
        # .config("spark.submit.deployMode", "client")
        .getOrCreate()
    )
    return spark

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