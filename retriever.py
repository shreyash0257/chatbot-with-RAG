from rank_bm25 import BM25Okapi

def build_bm25_index(chunks):
    """Builds a BM25 index from a list of text chunks."""
    # Simple tokenization: lowercase and split by whitespace
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)

def dense_search(vectorstore, query, k):
    """
    Performs dense similarity search using Chroma.
    Returns a list of tuples: [(chunk_text, score), ...]
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    scored_results = []
    for doc, dist in results:
        # Distance metric varies, we just use rank later anyway,
        # but to have a score we invert the distance.
        score = 1.0 / (1.0 + dist)
        scored_results.append((doc.page_content, score))
    return scored_results

def sparse_search(bm25_index, chunks, query, k):
    """
    Performs sparse search using BM25.
    Returns a list of tuples: [(chunk_text, score), ...]
    """
    tokenized_query = query.lower().split()
    doc_scores = bm25_index.get_scores(tokenized_query)
    
    # Pair up chunks with scores and sort
    scored_chunks = list(zip(chunks, doc_scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return scored_chunks[:k]

def reciprocal_rank_fusion(dense_results, sparse_results, k=60, alpha=0.5):
    """
    Combines dense and sparse results using a weighted Reciprocal Rank Fusion.
    """
    # Create rank dictionaries mapping chunk_text -> rank (1-indexed)
    dense_ranks = {chunk: rank for rank, (chunk, _) in enumerate(dense_results, 1)}
    sparse_ranks = {chunk: rank for rank, (chunk, _) in enumerate(sparse_results, 1)}
    
    all_chunks = set(dense_ranks.keys()) | set(sparse_ranks.keys())
    
    rrf_scores = []
    for chunk in all_chunks:
        dense_rank = dense_ranks.get(chunk, 1000)
        sparse_rank = sparse_ranks.get(chunk, 1000)
        
        dense_score = 1.0 / (dense_rank + k)
        sparse_score = 1.0 / (sparse_rank + k)
        
        final_score = (alpha * dense_score) + ((1.0 - alpha) * sparse_score)
        rrf_scores.append((chunk, final_score))
        
    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, score in rrf_scores]

def hybrid_search(vectorstore, bm25_index, chunks, query, top_k=20, alpha=0.5):
    """
    Executes hybrid search (dense + sparse) and fuses results.
    Returns a list of chunks.
    """
    if alpha == 1.0 or bm25_index is None:
        dense_results = dense_search(vectorstore, query, top_k)
        return [chunk for chunk, _ in dense_results]
    elif alpha == 0.0:
        sparse_results = sparse_search(bm25_index, chunks, query, top_k)
        return [chunk for chunk, _ in sparse_results]
        
    dense_results = dense_search(vectorstore, query, top_k)
    sparse_results = sparse_search(bm25_index, chunks, query, top_k)
    
    fused_chunks = reciprocal_rank_fusion(dense_results, sparse_results, alpha=alpha)
    return fused_chunks[:top_k]
