# config.py

# Feature Flags
HYBRID_SEARCH_ENABLED = True
RERANKING_ENABLED = True

# Parameters
HYBRID_ALPHA = 0.5           # weight for dense vs sparse (0=sparse only, 1=dense only)
RETRIEVAL_TOP_K = 20         # how many candidates to fetch before reranking
RERANKER_TOP_N = 5           # how many to pass to LLM after reranking
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
