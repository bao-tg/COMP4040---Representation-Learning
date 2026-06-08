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

For server-side downloads, omit `--with-pretrained-w2v` and/or `--with-glove`,
then use `--download-pretrained-w2v` and/or `--download-glove` during the
embedding stage. The GloVe download pulls `glove.840B.300d.zip` and extracts
`glove.840B.300d.txt`, so run it inside tmux.

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

For long runs, start a tmux session first:

```bash
tmux new -s comp4040-runtime
cd ~/COMP4040---Representation-Learning
source .venv/bin/activate
```

Detach without stopping the run with `Ctrl-b` then `d`. Reattach with:

```bash
tmux attach -t comp4040-runtime
```

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
  --pretrained-w2v-path GoogleNews-vectors-negative300.bin \
  --glove-path glove.840B.300d.txt \
  --transformer-batch-size 32
```

Build all embeddings and download both pretrained static-vector files if they
are missing:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf,w2v,glove,sbert,bge \
  --download-pretrained-w2v \
  --download-glove \
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

Download pretrained GloVe 840B 300D from Stanford:

```bash
python scripts/run_server_full.py \
  --stage embeddings \
  --encoders glove \
  --download-glove \
  --force-embeddings
```

Train corpus-specific Word2Vec and GloVe on the cleaned Amazon review corpus:

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

This writes `data/embeddings/w2v_trained.npy` and
`data/embeddings/glove_trained.npy`. It does not replace the pretrained
`w2v.npy` or `glove.npy` files.

Every embedding stage updates `data/embeddings/embedding_run_summary.json` with
per-encoder internal timing. For report tables, still prefer the separate
`/usr/bin/time -v` commands in Section 5 because they produce auditable logs for
each encoder.

Run all analysis tasks:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,sbert,bge \
  --analysis-tasks all \
  --umap-sample-size 50000 \
  --retrieval-queries 5000
```

Run the same analysis including corpus-trained static encoders:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,w2v_trained,glove_trained,sbert,bge \
  --analysis-tasks all \
  --umap-sample-size 50000 \
  --retrieval-queries 5000
```

`--force-embeddings` prevents reuse of an old same-shape `w2v.npy` during W2V
replacement.

The analysis stage always validates the processed parquet and selected
embeddings before running.

## 5. Runtime Reruns For Report Tables

If report tables need measured embedding runtimes, wrap selected embedding
commands with `/usr/bin/time -v` and keep the logs. The efficiency summary can
parse `/usr/bin/time -v` elapsed lines and falls back to the per-encoder runtime
stored in `data/embeddings/embedding_run_summary.json`.

TF-IDF, Word2Vec, and GloVe are CPU-bound even when they run on a GPU server.
SBERT and BGE use CUDA through SentenceTransformer when CUDA is available.

```bash
mkdir -p logs

PYTHONUNBUFFERED=1 /usr/bin/time -v .venv/bin/python scripts/run_server_full.py \
  --stage embeddings \
  --encoders tfidf \
  --force-embeddings \
  2>&1 | tee logs/tfidf_runtime_$(date +%Y%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 /usr/bin/time -v .venv/bin/python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v \
  --pretrained-w2v-path GoogleNews-vectors-negative300.bin \
  --force-embeddings \
  2>&1 | tee logs/w2v_runtime_$(date +%Y%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 /usr/bin/time -v .venv/bin/python scripts/run_server_full.py \
  --stage embeddings \
  --encoders glove \
  --glove-path glove.840B.300d.txt \
  --download-glove \
  --force-embeddings \
  2>&1 | tee logs/glove_runtime_$(date +%Y%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 /usr/bin/time -v .venv/bin/python scripts/run_server_full.py \
  --stage embeddings \
  --encoders sbert \
  --force-embeddings \
  --transformer-batch-size 32 \
  2>&1 | tee logs/sbert_runtime_$(date +%Y%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 /usr/bin/time -v .venv/bin/python scripts/run_server_full.py \
  --stage embeddings \
  --encoders w2v_trained \
  --trained-w2v-epochs 5 \
  --force-embeddings \
  2>&1 | tee logs/w2v_trained_runtime_$(date +%Y%m%d_%H%M%S).log

PYTHONUNBUFFERED=1 /usr/bin/time -v .venv/bin/python scripts/run_server_full.py \
  --stage embeddings \
  --encoders glove_trained \
  --trained-glove-epochs 25 \
  --trained-glove-max-vocab 50000 \
  --force-embeddings \
  2>&1 | tee logs/glove_trained_runtime_$(date +%Y%m%d_%H%M%S).log
```

Refresh the efficiency and final summary after runtime reruns:

```bash
python scripts/run_server_full.py \
  --stage analysis \
  --encoders tfidf,w2v,glove,w2v_trained,glove_trained,sbert,bge \
  --analysis-tasks efficiency,summary
```

## 6. Analysis-Only Reruns

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
News file instead of gensim download. `--glove-path glove.840B.300d.txt` uses
the local GloVe file instead of downloading the Stanford zip.

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

## 7. Output Locations

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

## 8. Pull Artifacts Back To Local

From the local checkout, pull report artifacts from the server:

```bash
REMOTE=<ssh-alias> scripts/pull_server_artifacts.sh
```

This pulls `experiments/server_full/`, `logs/`, processed summary JSON, and
embedding metadata. To also pull the large `.npy` embedding arrays:

```bash
REMOTE=<ssh-alias> scripts/pull_server_artifacts.sh --with-embeddings
```

## 9. Resume And Force Rules

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
