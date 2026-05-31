# Representation Learning Project Study Guide

Concepts, experiments, metrics, and final interpretation for the COMP4040
Amazon Reviews representation-learning project.

## Repository Layers

- Source: notebooks, scripts, and `src/` modules.
- Artifacts: ignored local/server outputs under `data/`, `experiments/`, and
  `logs/`.

Main summary table:

```text
experiments/server_full/summary/final_analysis_summary.md
```

## 1. Dataset And Preprocessing

The full run uses five Amazon review categories with 400,000 raw rows each. The
streaming preprocessing stage cleans text, removes empty or very short reviews,
deduplicates cross-chunk records, and writes a single Parquet file.

Preprocessing properties:

- Streaming preprocess avoids loading all raw JSONL rows into memory.
- Parquet is columnar and efficient for analytics workloads.
- Text is normalized, whitespace-cleaned, truncated to a maximum token count,
  and stored as `cleaned_text`.
- The final full run keeps `1,946,618` rows from `2,000,000` raw rows.

## 2. Text Encoders

The project compares five representation families. The static embedding
baselines use pretrained external word vectors for both Word2Vec and GloVe.

### TF-IDF + TruncatedSVD

TF-IDF creates sparse lexical vectors based on term importance. TruncatedSVD
compresses the sparse matrix into dense vectors, similar to Latent Semantic
Analysis.

Strengths:

- Fast baseline.
- Good for lexical similarity and category-specific vocabulary.

Limitations:

- Does not capture contextual meaning well.
- Semantically similar texts with different wording may look far apart.

### Word2Vec

Word2Vec learns static word embeddings from local context windows. The project
uses pretrained Google News Word2Vec 300D vectors, then turns review text into
document vectors by averaging token vectors.

Strengths:

- Provides an external pretrained static embedding baseline.
- Lightweight compared with transformer encoders.

Limitations:

- Static embeddings cannot disambiguate word senses by context.
- Simple averaging loses word order and sentence structure.
- Requires the external Google News Word2Vec vectors, either as a local
  `GoogleNews-vectors-negative300.bin` file or via gensim download/cache.

### GloVe

GloVe uses pretrained global word co-occurrence vectors. Like Word2Vec, the
project averages token vectors to produce review embeddings.

Strengths:

- Strong static baseline with external semantic knowledge.
- Comparable to pretrained Word2Vec as a static external baseline.

Limitations:

- Still static and context-insensitive.
- Requires the external `glove.840B.300d.txt` file.
- This run uses pretrained GloVe; it does not train GloVe from scratch on the
  Amazon corpus.

### SBERT

SBERT produces sentence embeddings optimized for semantic similarity. The
project uses `all-MiniLM-L6-v2`.

Strengths:

- Captures sentence-level meaning.
- Strong clustering behavior in the final run.

Limitations:

- More expensive than static methods.
- Smaller than BGE-large, so it can miss some fine-grained semantic signal.

### BGE-large

BGE-large (`BAAI/bge-large-en-v1.5`) is a transformer embedding model trained
for high-quality semantic retrieval.

Strengths:

- Best retrieval and linear probe performance in the final run.
- Strong contextual representation.

Limitations:

- Expensive to encode at full scale.
- GPU is strongly preferred for full 2M-row runs.

## 3. Experiment 1: Clustering

Goal: measure whether unsupervised clusters align with product categories.

Algorithm:

- MiniBatchKMeans, scalable to full embeddings.
- Number of clusters equals number of category labels.

Metrics:

- NMI: normalized mutual information between clusters and categories.
- ARI: adjusted pairwise agreement between predicted clusters and categories.
- Silhouette: internal cohesion/separation score, computed on a sample for
  scalability.

Visualization:

- UMAP uses a 50,000-point stratified sample for visual comparison.
- The project saves both category-colored and KMeans-cluster-colored figures.

Final interpretation:

- SBERT has the strongest category-aligned clustering.
- BGE is also strong but performs best on retrieval and linear probe instead.

## 4. Experiment 2: Semantic Retrieval

Goal: evaluate whether similar-category reviews are near each other in embedding
space.

Algorithm:

- FAISS exact L2 index over the full embedding matrix.
- `--retrieval-queries 5000` means 5,000 sampled query reviews are evaluated,
  while the index still contains all full-run rows.

Metrics:

- Precision@5, Precision@10, Precision@50.
- MRR: mean reciprocal rank of the first relevant retrieved item.

Final interpretation:

- BGE-large is the strongest semantic retrieval encoder.
- SBERT is second.
- Compare GloVe directly against W2V as the pretrained-static baseline pair.
- The switch from self-trained W2V to pretrained Google News W2V makes the W2V
  comparison more consistent with GloVe, because both are now external static
  word-vector baselines.

## 5. Experiment 3: Anomaly Detection

Goal: surface reviews that are unusual in embedding space or inconsistent
between text and star rating.

Methods:

- Isolation Forest flags the top 1% embedding-space outliers.
- Rating-text inconsistency uses a scalable L2 `SGDRegressor` to predict star
  ratings from frozen embeddings and flags the top residuals.

Important caveat:

- The anomaly count is controlled by `contamination=0.01`, so counts are not an
  encoder ranking. Interpret anomaly CSV examples and residual thresholds
  qualitatively.

## 6. Experiment 4: Linear Probe

Goal: test how much rating-relevant information is encoded in frozen
representations.

Method:

- Train a lightweight L2 logistic probe with `SGDClassifier(loss="log_loss")`.
- Target labels are 1-5 star ratings.
- Stratified 80/20 train/test split.

Metrics:

- Accuracy.
- Macro-F1.
- Weighted-F1.
- Per-class F1 and confusion matrix.

Final interpretation:

- BGE-large is best by accuracy and F1, meaning its embeddings preserve the most
  rating-relevant signal.

## 7. Experiment 5: Efficiency And Scalability

Goal: document the cost of full-scale embedding generation and analysis.

Reported fields:

- Rows and embedding dimensions.
- File size per embedding.
- Device used where available.
- Parsed encode runtime from logs, especially for BGE-large.

Key point:

- Embedding generation happens once and is reused through `.npy` files and
  `.meta.json` completion metadata.

## 8. Final Result Summary

Final full-run artifacts are saved under:

```text
experiments/server_full/
```

The main summary table is:

```text
experiments/server_full/summary/final_analysis_summary.md
```

High-level conclusion:

- BGE-large is best for semantic retrieval and rating prediction.
- SBERT is best for category-aligned clustering.
- GloVe and W2V are both pretrained static embedding baselines.
- TF-IDF remains a useful lexical baseline.
- Word2Vec is lightweight but weaker on clustering in this setup.

W2V interpretation:

- Pretrained W2V improves nearest-neighbor retrieval compared with a small
  corpus-trained static model because it brings broader external semantic
  knowledge.
- It can still underperform on clustering and rating prediction because Google
  News vectors are not tuned to Amazon review categories, star ratings, or
  review-style sentiment.
- This is an expected tradeoff between broad pretrained lexical semantics and
  domain/task-specific signal.

Saved JSON, CSV, and PNG artifacts are the source of truth for reporting.
