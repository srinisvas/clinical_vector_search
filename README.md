# clinical_vector_search: End-to-end pipeline for semantic search over clinical transcriptions (MTSamples) with:

  1) Baseline vector search (PySpark + Sentence-Transformers + FAISS)
  2) Differential Privacy (adds Gaussian noise to embeddings before indexing)
  3) FHE POC (query-only encryption with TenSEAL CKKS; tiny subset demo)
  4) RAG-faster enhancements (HNSW index + hybrid BM25+vector + MMR re-rank)

**Doker commands to setup**

docker-compose up --build -d
docker exec -it <spark-worker-container-name> bash (for local run)


**Commands to run**
  ## Baseline pipeline: build FAISS and run a sample query
  python clinical_vector_search.py --mode baseline \
      --data_path /path/to/medical_transcriptions.csv \
      --index_path ./faiss_mtsamples_baseline.faiss

  ## Differential Privacy (0.15): build noisy index + compare vs baseline
  python clinical_vector_search.py --mode dp \
      --data_path /path/to/medical_transcriptions.csv \
      --index_path ./faiss_mtsamples_dp_sigma015.faiss \
      --sigma 0.15

  ## FHE POC (tiny subset; query-only encryption)
  python clinical_vector_search.py --mode fhe \
      --data_path /path/to/medical_transcriptions.csv \
      --subset 300

  ## RAG-faster (HNSW + hybrid + MMR)
  python clinical_vector_search.py --mode rag \
      --data_path /path/to/medical_transcriptions.csv \
      --index_path ./faiss_mtsamples_hnsw.faiss \
      --enable_hybrid \
      --bm25_topk 100 \
      --mmr_lambda 0.6

Dependencies (pip):
  pyspark sentence-transformers faiss-cpu numpy pandas scikit-learn rank-bm25
Optional:
  tenseal  (for FHE POC)
