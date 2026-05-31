# COMP4040 Representation Learning

This repository contains a reproducible representation-learning pipeline for
Amazon review text. It compares text encoders on clustering, semantic retrieval,
anomaly detection, linear probing, UMAP visualization, and runtime efficiency.

## Full-Run Artifacts

The full experiment has been run on the five-encoder setup. The generated
dataset, embeddings, logs, and experiment artifacts are local/server artifacts
and are intentionally not tracked in Git.

Summary artifacts:

```text
experiments/server_full/summary/final_analysis_summary.md
experiments/server_full/summary/final_analysis_summary.csv
experiments/server_full/summary/final_analysis_summary.json
```

The full processed corpus contains `1,946,618` cleaned review rows from
`2,000,000` raw rows across five Amazon review categories.

## Encoders

- TF-IDF + TruncatedSVD
- pretrained Word2Vec Google News 300D (`w2v`)
- GloVe 840B 300D (`glove`)
- SBERT `all-MiniLM-L6-v2`
- BGE-large `BAAI/bge-large-en-v1.5`

## What Is Tracked

Commit source, notebooks, scripts, requirements, and docs:

```text
README.md
StudyGuide.md
SERVER_RUNBOOK.md
.env.example
requirements.txt
notesbook/
scripts/
src/
```

Do not commit generated or local-only files:

```text
.env
data/
experiments/
logs/
glove.840B.300d.txt
GoogleNews-vectors-negative300.bin
```

The full dataset, embeddings, and result artifacts are intentionally ignored by
Git because they are large and machine-specific. They can be regenerated with
the commands below.

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

## Data

Place raw Amazon review JSONL files under:

```text
data/raw/
```

The full pipeline discovers all matching raw files and streams them into:

```text
data/processed/cleaned_reviews.parquet
```

External pretrained static-vector files are optional and not committed:

```text
glove.840B.300d.txt                 # required to rebuild glove.npy
GoogleNews-vectors-negative300.bin  # required to rebuild w2v.npy from a local file
```

The Word2Vec model can also be downloaded into the local gensim cache by
passing `--download-pretrained-w2v`.

The current report setup treats both Word2Vec and GloVe as pretrained static
baselines. It does not self-train Word2Vec on the Amazon corpus.

## Main Full-Run Commands

Preprocess raw JSONL:

```bash
python scripts/run_server_full.py --stage preprocess --force-preprocess
```

Build embeddings:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf,w2v,glove,sbert,bge \
  --download-pretrained-w2v \
  --transformer-batch-size 32
```

Rebuild only the Word2Vec embedding from pretrained Google News vectors:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v \
  --download-pretrained-w2v \
  --force-embeddings
```

Using a local Google News file:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v \
  --pretrained-w2v-path GoogleNews-vectors-negative300.bin \
  --force-embeddings
```

Run the full analysis suite without rebuilding embeddings:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks all \
  --umap-sample-size 50000 \
  --retrieval-queries 5000
```

To redraw UMAP with existing coordinates:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks umap,summary \
  --umap-sample-size 50000
```

`--force-umap` recomputes UMAP coordinates instead of redrawing from existing
coordinate files.

## Output Layout

Generated outputs are written under:

```text
data/processed/
data/embeddings/
data/models/
experiments/server_full/
logs/
```

The final summary table is:

```text
experiments/server_full/summary/final_analysis_summary.md
```

UMAP comparison figures are:

```text
experiments/server_full/figures/all_encoders_umap_category_grid.png
experiments/server_full/figures/all_encoders_umap_cluster_grid.png
```

## Notebook Workflow

The notebooks mirror the pipeline stages and are useful for explanation and
interactive checks:

```text
notesbook/01_eda.ipynb
notesbook/02_preprocess.ipynb
notesbook/03_encode.ipynb
notesbook/04_clustering.ipynb
notesbook/05_retrieval.ipynb
notesbook/06_anomaly.ipynb
```

`RUN_MODE = "sample"` is for quick local checks. `RUN_MODE = "full"` is for the
full dataset. Long full runs use `scripts/run_server_full.py` because it is
resumable and validates existing artifacts.

## Project Structure

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
scripts/run_server_sample.py# sample/smoke pipeline entrypoint
```

See `SERVER_RUNBOOK.md` for server deployment and long-run instructions.
See `StudyGuide.md` for experiment definitions, metrics, and interpretation.
