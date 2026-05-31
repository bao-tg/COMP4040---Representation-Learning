# COMP4040 Server Runbook

This runbook is for long full-data runs on a remote server. Set an SSH alias
for your own machine, then point `REMOTE` and `REMOTE_DIR` at that server. The
examples below assume the remote repo lives at:

```text
~/COMP4040---Representation-Learning
```

## 1. Sync To Server

From the local repo:

```bash
cd /path/to/COMP4040---Representation-Learning
REMOTE=<ssh-alias> scripts/deploy_home_server.sh full --clean-raw --with-glove --with-pretrained-w2v
```

`--with-glove` includes the local `glove.840B.300d.txt` file for rebuilding
GloVe embeddings.
`--with-pretrained-w2v` includes the local
`GoogleNews-vectors-negative300.bin` file for rebuilding W2V embeddings.

For gensim-cache download on the server, omit `--with-pretrained-w2v` and use
`--download-pretrained-w2v` during the embedding stage.

For source-only sync:

```bash
scripts/deploy_home_server.sh code --bootstrap
```

## 2. Enter The Server Environment

```bash
ssh <ssh-alias>
cd ~/COMP4040---Representation-Learning
source .venv/bin/activate
```

Bootstrap missing environment:

```bash
bash scripts/bootstrap_server.sh
source .venv/bin/activate
```

## 3. Validate GPU For Transformer Runs

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

BGE-large is practical on GPU and very slow on CPU. TF-IDF, Word2Vec, and GloVe
do not need CUDA.

## 4. Full Pipeline

Run the full pipeline in resumable stages. Do not rerun embeddings if existing
`.npy` and `.meta.json` artifacts validate correctly.

Preprocess all raw JSONL files:

```bash
python scripts/run_server_full.py --stage preprocess --force-preprocess
```

Build all embeddings:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf,w2v,glove,sbert,bge \
  --download-pretrained-w2v \
  --transformer-batch-size 32
```

Rebuild only W2V from pretrained Word2Vec:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v \
  --pretrained-w2v-path GoogleNews-vectors-negative300.bin \
  --force-embeddings
```

Download pretrained W2V through gensim:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v \
  --download-pretrained-w2v \
  --force-embeddings
```

Run all analysis tasks:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks all \
  --umap-sample-size 50000 \
  --retrieval-queries 5000
```

`--force-embeddings` prevents reuse of an old same-shape `w2v.npy` during W2V
replacement.

The analysis stage always validates the processed parquet and selected
embeddings before running.

## 5. Analysis-Only Reruns

Replace only the W2V embedding with pretrained Word2Vec and refresh W2V
analysis while reusing the other four encoders:

```bash
rm -f data/embeddings/w2v.npy \
      data/embeddings/w2v.npy.meta.json \
      data/models/w2v.model \
      experiments/server_full/umap/w2v_umap_coords.npy

python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v \
  --download-pretrained-w2v \
  --force-embeddings

python scripts/run_server_full.py \
  --stage analysis \
  --encoders w2v \
  --analysis-tasks clustering,retrieval,anomaly,linear_probe,efficiency \
  --retrieval-queries 5000

python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks umap,summary \
  --umap-sample-size 50000
```

`--pretrained-w2v-path GoogleNews-vectors-negative300.bin` uses the local Google
News file instead of gensim download.

Run only selected tasks without touching embeddings:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks umap,linear_probe,efficiency,summary \
  --umap-sample-size 50000
```

Redraw UMAP from existing coordinates:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks umap,summary \
  --umap-sample-size 50000
```

Recompute UMAP coordinates instead of reusing them:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks umap,summary \
  --umap-sample-size 50000 \
  --force-umap
```

## 6. Output Locations

```text
data/processed/cleaned_reviews.parquet
data/processed/cleaned_reviews_summary.json
data/embeddings/*.npy
data/embeddings/*.meta.json
data/models/
experiments/server_full/
logs/
```

Key report artifacts:

```text
experiments/server_full/summary/final_analysis_summary.md
experiments/server_full/clustering/all_clustering_metrics.json
experiments/server_full/retrieval/all_retrieval_metrics.json
experiments/server_full/anomaly/all_anomaly_metrics.json
experiments/server_full/linear_probe/all_linear_probe_metrics.json
experiments/server_full/umap/all_umap_summary.json
experiments/server_full/figures/all_encoders_umap_category_grid.png
experiments/server_full/figures/all_encoders_umap_cluster_grid.png
```

## 7. Resume And Force Rules

Embeddings are reused when their `.npy` shape and `.meta.json` match the
processed row count.

Force rebuild processed data:

```bash
python scripts/run_server_full.py --stage preprocess --force-preprocess
```

Force rebuild selected embeddings:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders bge \
  --force-embeddings
```

`--stage all` runs preprocess, embeddings, and analysis in one command. Explicit
stages are safer for long server jobs.
