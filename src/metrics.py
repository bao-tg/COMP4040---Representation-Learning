import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score

def compute_clustering_metrics(X, labels_true, labels_pred, silhouette_sample_size=20000, random_state=42):
    """
    Computes clustering metrics: NMI, ARI, and Silhouette Score.
    
    Parameters:
    - X: The feature matrix (embeddings).
    - labels_true: The ground truth labels (e.g., categories or ratings).
    - labels_pred: The predicted cluster labels.
    - silhouette_sample_size: Number of samples to use for Silhouette to avoid OOM on large datasets.
    
    Returns:
    - Dictionary with 'nmi', 'ari', and 'silhouette' scores.
    """
    metrics = {}
    
    print("Computing NMI and ARI...")
    metrics['nmi'] = normalized_mutual_info_score(labels_true, labels_pred)
    metrics['ari'] = adjusted_rand_score(labels_true, labels_pred)
    
    print(f"Computing Silhouette Score (sampled to {silhouette_sample_size} points)...")
    # If the dataset is smaller than the sample size, use the whole dataset
    sample_size = min(silhouette_sample_size, X.shape[0])
    
    # We only compute Silhouette if there's more than 1 cluster
    unique_labels = np.unique(labels_pred)
    if len(unique_labels) > 1 and len(unique_labels) < X.shape[0]:
        metrics['silhouette'] = silhouette_score(
            X, 
            labels_pred, 
            sample_size=sample_size, 
            random_state=random_state
        )
    else:
        print("Warning: Only 1 cluster found, Silhouette score is undefined.")
        metrics['silhouette'] = -1.0 # Or NaN
        
    return metrics
