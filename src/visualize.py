import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import umap
import os

def plot_umap_2d(X_high_dim, labels, title, filename=None, sample_size=10000, random_state=42):
    """
    Reduces embeddings to 2D using UMAP and plots them, colored by labels.
    Uses a sample to keep plotting fast and readable.
    """
    print(f"Generating UMAP 2D projection for {title}...")
    
    # Subsample for visualization to avoid overplotting and save time
    if X_high_dim.shape[0] > sample_size:
        np.random.seed(random_state)
        indices = np.random.choice(X_high_dim.shape[0], sample_size, replace=False)
        X_sample = X_high_dim[indices]
        labels_sample = np.array(labels)[indices]
    else:
        X_sample = X_high_dim
        labels_sample = labels

    reducer = umap.UMAP(n_components=2, random_state=random_state)
    embedding_2d = reducer.fit_transform(X_sample)
    
    plt.figure(figsize=(10, 8))
    
    # Check if labels are discrete or continuous
    if len(np.unique(labels_sample)) <= 20:
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_sample, palette="tab10", s=10, alpha=0.7)
    else:
        sns.scatterplot(x=embedding_2d[:, 0], y=embedding_2d[:, 1], hue=labels_sample, palette="viridis", s=10, alpha=0.7)
        
    plt.title(f'UMAP Projection: {title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if filename:
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filename}")
    
    plt.show()

def plot_metrics_comparison(metrics_dict, filename=None):
    """
    Plots a bar chart comparing metrics across different encoders.
    
    metrics_dict format: 
    {
        'TF-IDF': {'nmi': 0.5, 'ari': 0.4, 'silhouette': 0.1},
        'SBERT': {'nmi': 0.7, 'ari': 0.6, 'silhouette': 0.3},
        ...
    }
    """
    print("Plotting metrics comparison...")
    encoders = list(metrics_dict.keys())
    
    # Get all unique metric names
    metric_names = set()
    for m in metrics_dict.values():
        metric_names.update(m.keys())
    metric_names = sorted(list(metric_names))
    
    # Prepare data for seaborn
    data = []
    for encoder, metrics in metrics_dict.items():
        for m_name, m_val in metrics.items():
            data.append({'Encoder': encoder, 'Metric': m_name.upper(), 'Score': m_val})
            
    import pandas as pd
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Metric', y='Score', hue='Encoder', palette='Set2')
    plt.title('Clustering Performance Comparison across Encoders')
    plt.ylim(0, 1.0) # Assuming most metrics are 0-1. Silhouette can be -1 to 1, but usually positive here.
    
    if filename:
        dirname = os.path.dirname(filename)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {filename}")
        
    plt.show()
