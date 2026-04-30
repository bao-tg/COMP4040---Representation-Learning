# Representation Learning Project: Study Guide & Key Concepts

Based on the codebase we've built, here is a structured summary of the key theoretical concepts, algorithms, and evaluation metrics you should study to deeply understand and defend this project.

---

## 1. Data Processing & Sampling
Before modeling, you handle massive amounts of text data efficiently.
* **Reservoir Sampling**: An algorithm used in `download_amazon_reviews.py`. It allows you to select a random, uniform sample from a stream of data (like massive HuggingFace datasets) without needing to load the entire dataset into memory.
* **Stratified Sampling**: Ensuring that your final dataset has an equal representation of classes (e.g., exactly 20% for each 1-5 star rating) to prevent class imbalance from skewing your models.
* **Parquet Format**: A columnar storage file format. You should understand why it is much faster and more compressed for data analytics compared to CSV or JSON.

---

## 2. Text Representation (Encoders)
You implemented 5 different ways to convert text into mathematical vectors. You must understand the evolution and differences between them:

### Sparse & Linear Models
* **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure evaluating how relevant a word is to a document. It creates sparse, high-dimensional vectors.
* **Truncated SVD (Singular Value Decomposition)**: A linear dimensionality reduction technique. When applied to TF-IDF, it is known as **Latent Semantic Analysis (LSA)**. It extracts underlying topics and makes the sparse vectors dense.

### Static Word Embeddings
* **Word2Vec**: You should know the difference between its two architectures: *CBOW* (predicting a word given context) and *Skip-gram* (predicting context given a word). It captures local, linear semantic relationships (e.g., King - Man + Woman = Queen).
* **GloVe (Global Vectors)**: Unlike Word2Vec which uses local context windows, GloVe factorizes a global word co-occurrence matrix. 
* **Limitation**: Both are "static," meaning the word "bank" (river) and "bank" (money) get the exact same vector. You combined word vectors into document vectors via simple averaging.

### Contextual Sentence Embeddings
* **Transformer Architecture**: The underlying neural network architecture with self-attention mechanisms that powers modern NLP.
* **SBERT (Sentence-BERT)**: A modification of the BERT network using Siamese network structures to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity.
* **BGE (BAAI General Embedding)**: State-of-the-art models optimized heavily via *Contrastive Learning* to pull similar texts together and push dissimilar texts apart in the vector space. They solve the polysemy problem (handling multiple meanings of a word based on context).

---

## 3. Experiment 1: Clustering
*Goal: Grouping similar representations together without supervision.*

### Key Algorithms
* **Mini-Batch K-Means**: A variant of K-Means that updates cluster centroids using small random batches of data rather than the whole dataset. Essential for scaling to 1.5M records. Relies on Euclidean distance.
* **UMAP (Uniform Manifold Approximation and Projection)**: A modern, non-linear dimensionality reduction algorithm. Unlike PCA (which is linear), UMAP preserves local manifold structure and global relationships, making it excellent for visualizing high-dimensional clusters in 2D or 3D.

### Evaluation Metrics
* **NMI (Normalized Mutual Information)**: Measures the mutual dependence between the predicted clusters and ground truth labels. It is normalized between 0 (no mutual information) and 1 (perfect correlation).
* **ARI (Adjusted Rand Index)**: Computes a similarity measure between two clusterings by considering all pairs of samples. It is "adjusted" for chance, meaning a random clustering will score close to 0.0.
* **Silhouette Score**: An internal evaluation metric (doesn't need ground truth labels). It measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). Ranges from -1 to 1. 

---

## 4. Experiment 2: Semantic Retrieval
*Goal: Finding the most semantically similar reviews to a given query.*

### Key Algorithms
* **FAISS (Facebook AI Similarity Search)**: A library for efficient similarity search and clustering of dense vectors. You should understand the difference between exact search (`IndexFlatL2`, `IndexFlatIP`) and approximate nearest neighbor (ANN) search (like HNSW or IVF) which FAISS specializes in for billions of vectors.
* **L2 Distance vs. Cosine Similarity**: How geometric distance relates to the angle between two vectors.

### Evaluation Metrics
* **Precision@k**: The fraction of relevant documents among the top *k* retrieved documents. (e.g., If I retrieve 10 documents, and 3 have the correct category, Precision@10 is 0.3).
* **MRR (Mean Reciprocal Rank)**: A statistic evaluating responses to queries. The reciprocal rank of a query response is the multiplicative inverse of the rank of the *first* correct answer (e.g., if the first relevant document is at position 3, the score is 1/3). MRR is the average across all queries.

---

## 5. Experiment 3: Anomaly Detection
*Goal: Finding reviews that don't fit the expected patterns.*

### Key Algorithms
* **Isolation Forest**: An unsupervised algorithm that detects anomalies by isolating observations. It builds an ensemble of random decision trees; anomalies are isolated closer to the root of the tree (shorter path lengths) because they are fewer and different.
* **Ridge Regression (Tikhonov Regularization)**: Used to predict the star rating from the embeddings. You should understand how the L2 penalty (alpha) prevents overfitting on high-dimensional data.
* **Residual Analysis**: You detect anomalies by looking at the *residuals* (the absolute difference between the actual rating and the rating predicted by the text). High residuals indicate a semantic contradiction (e.g., sarcastic text with a contradictory star rating).
