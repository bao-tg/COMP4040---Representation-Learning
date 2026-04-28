import numpy as np
import umap
from sklearn.cluster import MiniBatchKMeans
from .metrics import compute_clustering_metrics

def reduce_dimensions_umap(X, n_components=64, random_state=42, sample_size=None):
    """
    Reduces the dimensionality of embeddings using UMAP.
    For very large datasets, UMAP can be extremely slow. 
    If sample_size is provided, it fits UMAP on a random subsample and transforms the rest.
    """
    # Cap n_components to be less than the number of samples for very small test datasets
    actual_n_components = min(n_components, X.shape[0] - 2)
    if actual_n_components < 2:
        actual_n_components = 2 # minimum for 2D plotting later if needed, though UMAP might still fail if N < 4
        
    print(f"Running UMAP to reduce dimensions to {actual_n_components}...")
    reducer = umap.UMAP(n_components=actual_n_components, random_state=random_state)
    
    if sample_size and X.shape[0] > sample_size:
        print(f"Subsampling to {sample_size} for UMAP fitting to save time/memory...")
        np.random.seed(random_state)
        indices = np.random.choice(X.shape[0], sample_size, replace=False)
        reducer.fit(X[indices])
        
        # Transform the whole dataset in batches to avoid memory spikes
        # Actually, umap transform can also be memory intensive, but let's try direct transform first
        print("Transforming all embeddings with fitted UMAP...")
        X_reduced = reducer.transform(X)
    else:
        print("Fitting and transforming UMAP on all data...")
        X_reduced = reducer.fit_transform(X)
        
    return X_reduced

def run_clustering_experiment(X, labels_true, k=10, reduce_dim=True, n_components=64, umap_sample_size=100000, random_state=42):
    """
    End-to-end clustering pipeline:
    1. Optional UMAP reduction.
    2. MiniBatchKMeans clustering.
    3. Evaluation.
    """
    if reduce_dim:
        X_clust = reduce_dimensions_umap(X, n_components=n_components, random_state=random_state, sample_size=umap_sample_size)
    else:
        X_clust = X
        
    print(f"Running MiniBatchKMeans with k={k}...")
    kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, batch_size=1024, n_init='auto')
    labels_pred = kmeans.fit_predict(X_clust)
    
    print("Evaluating clusters...")
    # Silhouette score is usually computed on the space the clustering was performed on.
    metrics = compute_clustering_metrics(X_clust, labels_true, labels_pred, random_state=random_state)
    
    return labels_pred, metrics, X_clust
