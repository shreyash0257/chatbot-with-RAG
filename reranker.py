from sentence_transformers import CrossEncoder
import streamlit as st

@st.cache_resource
def load_reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Loads and caches the cross-encoder model."""
    return CrossEncoder(model_name)

def rerank(reranker, query, chunks, top_n=5):
    """
    Reranks a list of chunks using a cross-encoder model.
    Returns a list of tuples: [(chunk_text, score), ...]
    """
    if not chunks:
        return []
        
    # Prepare pairs of (query, chunk)
    pairs = [[query, chunk] for chunk in chunks]
    
    # Score each pair
    scores = reranker.predict(pairs)
    
    # Combine chunks with scores and sort by score descending
    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N
    return scored_chunks[:top_n]
