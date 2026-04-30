import faiss
import numpy as np
from tqdm import tqdm
from .metrics import precision_at_k, mean_reciprocal_rank

def build_faiss_index(embeddings, index_type='L2'):
    """
    Builds a FAISS index from the given embeddings.
    For L2 distance, we use IndexFlatL2.
    For Cosine similarity, we normalize embeddings and use IndexFlatIP.
    """
    d = embeddings.shape[1]
    
    if index_type == 'Cosine':
        # FAISS uses Inner Product for Cosine similarity, but we need to normalize vectors first
        index = faiss.IndexFlatIP(d)
        faiss.normalize_L2(embeddings)
    else:
        # Default to L2 distance
        index = faiss.IndexFlatL2(d)
        
    print(f"Building FAISS {index_type} index with {embeddings.shape[0]} items...")
    index.add(embeddings)
    return index

def search_faiss_index(index, queries, k=10):
    """
    Searches the FAISS index for the given queries.
    Returns the distances and indices of the top k results.
    """
    distances, indices = index.search(queries, k)
    return distances, indices

def evaluate_retrieval(embeddings, labels, sample_size=5000, k_values=[5, 10, 50], random_state=42):
    """
    Evaluates semantic retrieval by sampling a subset of queries, searching the index,
    and calculating Precision@k and MRR based on label matching.
    """
    # FAISS requires float32 contiguous arrays
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # 1. Build Index
    index = build_faiss_index(embeddings, index_type='L2')
    
    # 2. Sample Queries
    np.random.seed(random_state)
    n_samples = min(sample_size, embeddings.shape[0])
    query_indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
    
    queries = embeddings[query_indices]
    query_labels = np.array(labels)[query_indices]
    
    # 3. Search
    max_k = max(k_values)
    # We retrieve max_k + 1 because the nearest neighbor to a query is usually the query itself
    print(f"Searching index for {n_samples} queries...")
    distances, indices = search_faiss_index(index, queries, k=max_k + 1)
    
    # 4. Evaluate
    print("Calculating metrics...")
    results = {'mrr': 0.0}
    for k in k_values:
        results[f'precision@{k}'] = 0.0
        
    for i in tqdm(range(n_samples), desc="Evaluating"):
        q_label = query_labels[i]
        
        # Remove the query itself from the results (index 0 is usually itself)
        # If it's not itself, we just take the top max_k
        pred_indices = indices[i]
        if pred_indices[0] == query_indices[i]:
            pred_indices = pred_indices[1:]
        else:
            pred_indices = pred_indices[:max_k]
            
        pred_labels = np.array(labels)[pred_indices]
        
        # Calculate Precision@k
        for k in k_values:
            results[f'precision@{k}'] += precision_at_k(q_label, pred_labels, k)
            
        # Calculate MRR
        results['mrr'] += mean_reciprocal_rank(q_label, pred_labels)
        
    # Average the results
    for key in results:
        results[key] /= n_samples
        
    return results
