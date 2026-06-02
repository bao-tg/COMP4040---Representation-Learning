# COMP4040 Representation Learning

This repository contains a reproducible representation-learning pipeline for
Amazon review text. It compares classical sparse/static-vector representations
and transformer sentence embeddings across clustering, retrieval, anomaly
detection, linear probing, UMAP visualization, and runtime efficiency.

## Project Status

The full five-encoder experiment has been completed on the full processed
corpus.

| Item | Status |
| --- | --- |
| Raw input scale | 2,000,000 Amazon review rows |
| Processed corpus | 1,946,618 cleaned rows |
| Encoders evaluated | TF-IDF, Word2Vec, GloVe, SBERT, BGE-large |
| Main run profile | `server_full` |
| Validation status | OK |
| Commit-safe report package | `reports/final/` |

Generated datasets, embeddings, raw experiment outputs, and logs are intentionally
kept out of Git. The sanitized report outputs under `reports/final/` are tracked
for review and report writing.

## Encoders

| Encoder | Representation | Dimension | Notes |
| --- | --- | ---: | --- |
| TF-IDF | TF-IDF + TruncatedSVD | 300 | Sparse lexical baseline reduced to dense vectors |
| W2V | Google News Word2Vec | 300 | Pretrained static word vectors averaged per review |
| GloVe | GloVe 840B | 300 | Pretrained static word vectors averaged per review |
| SBERT | `all-MiniLM-L6-v2` | 384 | SentenceTransformer encoder |
| BGE | `BAAI/bge-large-en-v1.5` | 1024 | Large SentenceTransformer encoder |

The current report treats Word2Vec and GloVe as pretrained static-vector
baselines; neither one is trained from scratch on the Amazon review corpus.

## Evaluation Tasks

| Task | Metrics / Outputs |
| --- | --- |
| Clustering | Normalized Mutual Information, Adjusted Rand Index, Silhouette |
| Retrieval | Mean Reciprocal Rank, Precision@5, Precision@10, Precision@50 |
| Anomaly detection | Isolation Forest anomaly count, rating inconsistency count |
| Linear probe | Accuracy, macro F1, weighted F1, confusion matrices |
| UMAP | Category-colored and cluster-colored 2D projections |
| Efficiency | embedding shape, file size, runtime, device/backend |

## Result Snapshot

Full precision tables and machine-readable metrics are available in
`reports/final/`.

| Encoder | Dim | Device | Embed Time | NMI | ARI | MRR | P@5 | Acc. | Macro F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 300 | CPU | 5:08.61 | 0.226 | 0.107 | 0.691 | 0.527 | 0.447 | 0.435 |
| W2V | 300 | CPU | 2:30.93 | 0.014 | 0.010 | 0.722 | 0.580 | 0.432 | 0.420 |
| GloVe | 300 | CPU | 1:56.52 | 0.030 | 0.023 | 0.742 | 0.597 | 0.433 | 0.408 |
| SBERT | 384 | CUDA | 3:01:27 | 0.410 | 0.348 | 0.819 | 0.718 | 0.456 | 0.443 |
| BGE | 1024 | CUDA | 65:56:44 | 0.342 | 0.294 | 0.828 | 0.730 | 0.590 | 0.584 |

Interpretation notes:

- BGE is strongest on retrieval and linear probing in the current full run.
- SBERT is strongest on clustering NMI and ARI.
- Static-vector baselines are much cheaper to embed, but are weaker on the main
  retrieval and linear-probe outcomes.
- Anomaly counts are controlled by the configured contamination threshold and
  should not be interpreted as a direct encoder quality ranking.

## Report Artifacts

Use these files for the final report:

```text
reports/final/final_analysis_summary.md
reports/final/final_analysis_summary.csv
reports/final/final_analysis_summary.json
reports/final/metrics/
reports/final/figures/
```

The report package intentionally excludes:

- full embedding arrays under `data/embeddings/`
- raw and processed review data
- runtime logs containing machine-specific paths
- anomaly row CSVs that include review text fields
- UMAP coordinate arrays and clustering assignment arrays

## Repository Structure

```text
src/full_pipeline.py        # full/sample streaming pipeline and analysis suite
src/preprocess.py           # text cleaning helpers
src/encoders.py             # encoder wrappers
src/cluster.py              # clustering helpers
src/retrieval.py            # retrieval helpers
src/anomaly.py              # anomaly detection helpers
src/metrics.py              # metric utilities
src/visualize.py            # plotting helpers
scripts/run_server_full.py  # resumable full pipeline entrypoint
scripts/run_server_sample.py # sample/smoke pipeline entrypoint
reports/final/              # commit-safe result package for report writing
notesbook/                  # explanatory notebooks mirroring the pipeline
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional Hugging Face token:

```bash
HF_TOKEN=...
```

## Data And External Models

Place raw Amazon review JSONL files under:

```text
data/raw/
```

The full pipeline streams all matching raw files into:

```text
data/processed/cleaned_reviews.parquet
```

External pretrained static-vector files are not committed:

```text
glove.840B.300d.txt                 # required to rebuild GloVe locally
GoogleNews-vectors-negative300.bin  # required to rebuild W2V from a local file
```

Both W2V and GloVe are pretrained file-backed encoders in this project. GloVe is
loaded from `glove.840B.300d.txt` at the repository root, and can be downloaded
with `--download-glove` if the file is missing. W2V is loaded from
`GoogleNews-vectors-negative300.bin` when `--pretrained-w2v-path` is provided,
and can also be downloaded through gensim with `--download-pretrained-w2v`.

## Reproduce The Full Pipeline

Preprocess raw JSONL files:

```bash
python scripts/run_server_full.py --stage preprocess --force-preprocess
```

Build embeddings:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf,w2v,glove,sbert,bge \
  --pretrained-w2v-path GoogleNews-vectors-negative300.bin \
  --glove-path glove.840B.300d.txt \
  --transformer-batch-size 32
```

If the pretrained static-vector files are missing, use explicit download flags:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf,w2v,glove,sbert,bge \
  --download-pretrained-w2v \
  --download-glove \
  --transformer-batch-size 32
```

`--download-glove` downloads and extracts the Stanford `glove.840B.300d.zip`
archive. This is intentionally opt-in because the zip is about 2 GB and the
extracted text file is much larger.

Run the full analysis suite without rebuilding embeddings:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks all \
  --umap-sample-size 50000 \
  --retrieval-queries 5000
```

Refresh only UMAP plots and summary from existing artifacts:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks umap,summary \
  --umap-sample-size 50000
```

Use `--force-embeddings` or `--force-umap` only when intentionally replacing
existing artifacts.

## Notebook Workflow

The notebooks mirror the pipeline stages and are useful for explanation,
inspection, and small interactive checks:

```text
notesbook/01_eda.ipynb
notesbook/02_preprocess.ipynb
notesbook/03_encode.ipynb
notesbook/04_clustering.ipynb
notesbook/05_retrieval.ipynb
notesbook/06_anomaly.ipynb
```

Use `RUN_MODE = "sample"` for quick local checks and `RUN_MODE = "full"` for the
full dataset. Long full runs should use `scripts/run_server_full.py` because it
is resumable and validates existing artifacts.

## Operational Notes

- `SERVER_RUNBOOK.md` contains server deployment, tmux, runtime logging, and
  artifact pull-back instructions.
- `StudyGuide.md` contains the study-oriented explanation of encoders, metrics,
  and interpretation.
- `reports/final/README.md` documents exactly which report artifacts are safe to
  commit.

## References

Dataset:

- Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. (2024). *Bridging
  Language and Items for Retrieval and Recommendation*. Amazon Reviews'23.
  https://amazon-reviews-2023.github.io/

Encoders and pretrained models:

- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation
  of Word Representations in Vector Space*. https://arxiv.org/abs/1301.3781
- Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global Vectors
  for Word Representation*. https://nlp.stanford.edu/projects/glove/
- Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks*. https://arxiv.org/abs/1908.10084
- SentenceTransformers. `all-MiniLM-L6-v2` model card.
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- BAAI. `bge-large-en-v1.5` model card.
  https://huggingface.co/BAAI/bge-large-en-v1.5

Methods and tooling:

- McInnes, L., Healy, J., & Melville, J. (2018). *UMAP: Uniform Manifold
  Approximation and Projection for Dimension Reduction*.
  https://arxiv.org/abs/1802.03426
- Johnson, J., Douze, M., & Jegou, H. (2017). *Billion-scale similarity search
  with GPUs*. https://arxiv.org/abs/1702.08734
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation Forest*.
  https://doi.org/10.1109/ICDM.2008.17
- scikit-learn documentation for TF-IDF, TruncatedSVD, clustering metrics,
  classification metrics, and IsolationForest. https://scikit-learn.org/stable/
