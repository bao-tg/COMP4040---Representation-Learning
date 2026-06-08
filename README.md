# COMP4040 Representation Learning

This repository contains a reproducible representation-learning pipeline for
large-scale Amazon review text. The main experiment compares five text
representations on the same cleaned corpus: TF-IDF/SVD, pretrained Word2Vec,
pretrained GloVe, SBERT, and BGE-large.

The final report focuses on category-aligned clustering, semantic retrieval,
UMAP visualization, runtime/storage efficiency, and a shallow linear probe for
five-class star-rating prediction. The codebase also keeps anomaly-diagnostic
utilities and a corpus-trained Word2Vec/GloVe ablation for additional analysis.

## Current Status

| Item | Status |
| --- | --- |
| Raw input scale | 2,000,000 Amazon review rows |
| Processed corpus | 1,946,618 cleaned review rows |
| Main encoders | TF-IDF, Word2Vec, GloVe, SBERT, BGE-large |
| Static-vector ablation | `w2v_trained`, `glove_trained` |
| Main run profile | `server_full` |
| Validation status | OK |
| Report source | `reports/docs/finalreport.tex` |
| Report PDF | `reports/docs/finalreport.pdf` |
| Commit-safe result package | `reports/final/` |

Generated data, embeddings, model files, experiment outputs, logs, caches, and
external vector files are intentionally ignored by Git. The tracked report
artifacts are the LaTeX report under `reports/docs/` and the sanitized result
package under `reports/final/`.

## Results Snapshot

Full precision tables and machine-readable metrics are in `reports/final/`.

| Encoder | Dim | Device | Embed Time | NMI | ARI | MRR | P@5 | Acc. | Macro F1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF | 300 | CPU | 5:08.61 | 0.226 | 0.107 | 0.691 | 0.527 | 0.447 | 0.435 |
| Word2Vec | 300 | CPU | 2:30.93 | 0.014 | 0.010 | 0.722 | 0.580 | 0.432 | 0.420 |
| GloVe | 300 | CPU | 1:56.52 | 0.030 | 0.023 | 0.742 | 0.597 | 0.433 | 0.408 |
| SBERT | 384 | CUDA | 3:01:27 | 0.410 | 0.348 | 0.819 | 0.718 | 0.456 | 0.443 |
| BGE-large | 1024 | CUDA | 65:56:44 | 0.342 | 0.294 | 0.828 | 0.730 | 0.590 | 0.584 |

Main interpretation:

- SBERT gives the strongest category-aligned clustering by NMI and ARI.
- BGE-large gives the strongest semantic retrieval and star-rating linear probe.
- TF-IDF remains a scalable lexical baseline and is stronger than static
  averaged word vectors for category clustering.
- Pretrained static word-vector averages are cheap and useful for local semantic
  retrieval, but they lose document-level composition.
- Anomaly outputs are auxiliary diagnostics. Their counts are affected by fixed
  thresholds and are not used as an encoder-quality leaderboard.

## Static-Vector Ablation

The report also compares pretrained Word2Vec/GloVe against corpus-trained
variants while keeping the same 300-dimensional averaged word-vector document
representation.

| Encoder | Token-vector source | NMI | ARI | MRR | P@5 | Acc. | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Word2Vec | Google News | 0.014 | 0.010 | 0.722 | 0.580 | 0.432 | 0.420 |
| Word2Vec | Amazon reviews | 0.143 | 0.119 | 0.752 | 0.613 | 0.492 | 0.481 |
| GloVe | Common Crawl 840B | 0.030 | 0.023 | 0.742 | 0.597 | 0.433 | 0.408 |
| GloVe | Amazon reviews | 0.007 | 0.004 | 0.618 | 0.430 | 0.308 | 0.286 |

This ablation is not a hardware-controlled runtime comparison. It was run after
the original server experiment and should be read as a representation-source
check: in-domain Word2Vec helps here, while in-domain GloVe does not.

## Repository Layout

```text
.
|-- src/
|   |-- full_pipeline.py      # authoritative full/sample pipeline implementation
|   |-- preprocess.py         # text cleaning helpers
|   |-- encoders.py           # lightweight encoder wrappers
|   |-- cluster.py            # clustering helpers
|   |-- retrieval.py          # retrieval helpers
|   |-- anomaly.py            # auxiliary anomaly diagnostics
|   |-- metrics.py            # metric utilities
|   `-- visualize.py          # plotting helpers
|-- scripts/
|   |-- run_server_full.py    # resumable full-data entrypoint
|   |-- run_server_sample.py  # sample/smoke entrypoint
|   |-- download_amazon_reviews.py
|   |-- deploy_home_server.sh
|   |-- bootstrap_server.sh
|   `-- pull_server_artifacts.sh
|-- notesbook/                # notebooks mirroring the pipeline stages
|-- reports/
|   |-- docs/                 # LaTeX report source, styles, figures, and PDF
|   `-- final/                # commit-safe metrics and report-ready figures
|-- data/                     # ignored raw/processed data, embeddings, models
|-- experiments/              # ignored full run outputs
|-- logs/                     # ignored runtime logs
|-- SERVER_RUNBOOK.md         # long-run server workflow
|-- StudyGuide.md             # study-oriented explanation
|-- requirements.txt
`-- README.md
```

Notes on cleanliness:

- `data/`, `experiments/`, `logs/`, `__pycache__/`, LaTeX intermediates, local
  vector files, and local environment files are ignored by `.gitignore`.
- `reports/final/` is the sanitized package intended for Git/report evidence.
- `reports/docs/finalreport.tex`, `reports/docs/finalreport.pdf`, report style
  files, references, and report figures are tracked.
- Large external vectors such as `GoogleNews-vectors-negative300.bin` and
  `glove.840B.300d.txt` are local-only dependencies.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional Hugging Face token:

```bash
cp .env.example .env
# then set HF_TOKEN=... if needed
```

## Data And External Models

Place sampled Amazon Reviews 2023 JSONL files under:

```text
data/raw/
```

The full preprocessing stage streams all matching raw files into:

```text
data/processed/cleaned_reviews.parquet
```

External pretrained static-vector files are not committed:

```text
GoogleNews-vectors-negative300.bin  # optional local Word2Vec file
glove.840B.300d.txt                 # optional local GloVe file
```

Word2Vec can use either a local Google News binary via `--pretrained-w2v-path`
or a gensim download via `--download-pretrained-w2v`. GloVe can use a local
`glove.840B.300d.txt` file or download/extract the Stanford archive with
`--download-glove`. The GloVe download is opt-in because the archive and
extracted text file are large.

## Reproduce The Full Pipeline

Preprocess raw JSONL files:

```bash
python scripts/run_server_full.py --stage preprocess --force-preprocess
```

Build the five main embeddings from local static-vector files:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf,w2v,glove,sbert,bge \
  --pretrained-w2v-path GoogleNews-vectors-negative300.bin \
  --glove-path glove.840B.300d.txt \
  --transformer-batch-size 32
```

Build the five main embeddings and allow downloads for missing static vectors:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf,w2v,glove,sbert,bge \
  --download-pretrained-w2v \
  --download-glove \
  --transformer-batch-size 32
```

Run the main analysis without rebuilding embeddings:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks clustering,retrieval,umap,linear_probe,efficiency,summary \
  --umap-sample-size 50000 \
  --retrieval-queries 5000
```

Use `--force-embeddings` only when intentionally replacing existing embedding
artifacts. Use `--force-umap` only when intentionally recomputing UMAP
coordinates.

## Run The Static-Vector Ablation

Build corpus-trained Word2Vec and GloVe embeddings:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v_trained,glove_trained \
  --trained-vector-size 300 \
  --trained-window 5 \
  --trained-min-count 5 \
  --trained-w2v-epochs 5 \
  --trained-glove-epochs 25 \
  --trained-glove-max-vocab 50000
```

Analyze the five main encoders plus the two corpus-trained variants:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,w2v_trained,glove_trained,sbert,bge \
  --analysis-tasks clustering,retrieval,linear_probe,efficiency,summary \
  --retrieval-queries 5000
```

The ablation writes separate artifacts:

```text
data/embeddings/w2v_trained.npy
data/embeddings/glove_trained.npy
data/models/w2v_trained.model
data/models/glove_trained.pt
```

These files are ignored and do not replace pretrained `w2v.npy` or `glove.npy`.

## Sample And Notebook Workflows

Run a small sample pipeline:

```bash
python scripts/run_server_sample.py \
  --rows-per-category 1000 \
  --encoders tfidf,w2v,sbert,bge \
  --download-pretrained-w2v
```

The notebooks mirror the pipeline stages for inspection and explanation:

```text
notesbook/01_eda.ipynb
notesbook/02_preprocess.ipynb
notesbook/03_encode.ipynb
notesbook/04_clustering.ipynb
notesbook/05_retrieval.ipynb
notesbook/06_anomaly.ipynb
```

For long full-data runs, prefer `scripts/run_server_full.py` over notebooks
because it is resumable and validates existing artifacts.

## Report Artifacts

Use these tracked files for report review and evidence:

```text
reports/docs/finalreport.tex
reports/docs/finalreport.pdf
reports/final/final_analysis_summary.md
reports/final/final_analysis_summary.csv
reports/final/final_analysis_summary.json
reports/final/metrics/
reports/final/figures/
```

`reports/final/` intentionally excludes raw data, full embedding arrays, local
runtime logs, anomaly row CSVs with review text fields, UMAP coordinate arrays,
and clustering assignment arrays.

## Operational Notes

- `SERVER_RUNBOOK.md` contains server deployment, tmux, runtime logging, and
  artifact pull-back instructions.
- `StudyGuide.md` contains a study-oriented explanation of encoders, metrics,
  and interpretation.
- `reports/final/README.md` documents the sanitized report package.

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
