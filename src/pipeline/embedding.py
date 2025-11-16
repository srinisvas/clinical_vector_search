from pyspark.sql import SparkSession, functions as F, types as T
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from pipeline.utils import normalize_rows, norm_vec


def build_spark(app_name="ClinicalVectorSearch"):
    """
    Build a Spark session in local mode.
    Works perfectly on macOS (Intel / M1 / M2 / M3 / M4).
    """

    print("Building Spark Session in local mode...")

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")  # local Spark, uses all CPU cores
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.driver.memory", "4g")            # safe default for mac
        .config("spark.executor.memory", "4g")
        .config("spark.driver.maxResultSize", "2g")
        .getOrCreate()
    )

    return spark


def embed_partition(texts: List[str], model_name: str) -> List[List[float]]:
    """
    Used only if you enable distributed inference.
    Model loads inside each Spark worker.
    """

    model = SentenceTransformer(model_name)
    vecs = model.encode(texts, batch_size=16, show_progress_bar=False)
    vecs = normalize_rows(np.array(vecs, dtype=np.float32))

    return vecs.tolist()


def add_noise_vec(vec: List[float], sigma: float) -> List[float]:
    """
    Differential privacy noise addition.
    """
    v = np.array(vec, dtype=np.float32)
    v = v + np.random.normal(0.0, sigma, size=v.shape).astype(np.float32)
    v = norm_vec(v)
    return v.tolist()
