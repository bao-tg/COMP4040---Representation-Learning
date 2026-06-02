#!/usr/bin/env bash
set -euo pipefail

MODE="sample"
REMOTE="${REMOTE:-your-server-alias}"
REMOTE_DIR="${REMOTE_DIR:-~/COMP4040---Representation-Learning}"
IDENTITY_FILE="${IDENTITY_FILE:-}"
SSH_OPTS=()
RSYNC_SSH="ssh"
if [[ -n "$IDENTITY_FILE" ]]; then
  SSH_OPTS=(-o IdentitiesOnly=yes -i "$IDENTITY_FILE")
  RSYNC_SSH="ssh -o IdentitiesOnly=yes -i $IDENTITY_FILE"
fi

usage() {
  cat <<'EOF'
Usage:
  scripts/deploy_home_server.sh [code|sample|full] [--bootstrap] [--clean-raw] [--with-glove] [--with-pretrained-w2v]

Modes:
  code    Sync source, notebooks, requirements only.
  sample  Sync source plus the existing small processed sample parquet.
  full    Sync source plus data/raw/*.jsonl for a server run.

Options:
  --bootstrap   Create/update the remote virtualenv after syncing.
  --clean-raw   Remove remote data/raw/*.jsonl before syncing full raw data.
  --with-glove  Include glove.840B.300d.txt. Omit unless running GloVe.
  --with-pretrained-w2v
                 Include GoogleNews-vectors-negative300.bin(.gz). Omit unless
                 running the pretrained W2V encoder from a local file.

Environment:
  REMOTE=your-server-alias
  REMOTE_DIR=~/COMP4040---Representation-Learning
  IDENTITY_FILE=/path/to/key  Optional. Omit to use normal SSH config.
EOF
}

BOOTSTRAP=0
CLEAN_RAW=0
WITH_GLOVE=0
WITH_PRETRAINED_W2V=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    code|sample|full)
      MODE="$1"
      ;;
    --bootstrap)
      BOOTSTRAP=1
      ;;
    --clean-raw)
      CLEAN_RAW=1
      ;;
    --with-glove)
      WITH_GLOVE=1
      ;;
    --with-pretrained-w2v)
      WITH_PRETRAINED_W2V=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
  shift
done

case "$MODE" in
  code|sample|full) ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if [[ -n "$IDENTITY_FILE" && ! -f "$IDENTITY_FILE" ]]; then
  echo "Missing SSH identity: $IDENTITY_FILE" >&2
  exit 1
fi

BASE_EXCLUDES=(
  "--exclude=.git/"
  "--exclude=.venv"
  "--exclude=.venv/"
  "--exclude=venv"
  "--exclude=venv/"
  "--exclude=env"
  "--exclude=env/"
  "--exclude=__pycache__/"
  "--exclude=*.pyc"
  "--exclude=.ipynb_checkpoints/"
  "--exclude=.pytest_cache/"
  "--exclude=.mypy_cache/"
  "--exclude=.vscode/"
  "--exclude=.env"
  "--include=.env.example"
  "--exclude=.env.*"
  "--exclude=logs/"
  "--exclude=experiments/"
  "--exclude=data/embeddings/"
  "--exclude=data/models/"
  "--exclude=glove.840B.300d.zip"
)

MODE_EXCLUDES=()
case "$MODE" in
  code)
    MODE_EXCLUDES=(
      "--exclude=data/"
      "--exclude=GoogleNews-vectors-negative300.bin"
      "--exclude=glove.840B.300d.txt"
      "--exclude=GoogleNews-vectors-negative300.bin.gz"
    )
    ;;
  sample)
    MODE_EXCLUDES=(
      "--exclude=data/raw/"
      "--exclude=GoogleNews-vectors-negative300.bin"
      "--exclude=glove.840B.300d.txt"
      "--exclude=GoogleNews-vectors-negative300.bin.gz"
      "--include=data/"
      "--include=data/processed/"
      "--include=data/processed/cleaned_reviews_sample_1000_per_category.parquet"
      "--include=data/processed/cleaned_reviews_sample_1000_per_category_summary.json"
      "--exclude=data/processed/*"
    )
    ;;
  full)
    MODE_EXCLUDES=(
      "--include=data/"
      "--include=data/raw/"
      "--include=data/raw/*.jsonl"
      "--include=data/processed/"
      "--include=data/processed/cleaned_reviews_sample_1000_per_category.parquet"
      "--include=data/processed/cleaned_reviews_sample_1000_per_category_summary.json"
      "--exclude=data/processed/*"
    )
    if [[ "$WITH_GLOVE" -eq 0 ]]; then
      MODE_EXCLUDES+=("--exclude=glove.840B.300d.txt")
    fi
    if [[ "$WITH_PRETRAINED_W2V" -eq 0 ]]; then
      MODE_EXCLUDES+=("--exclude=GoogleNews-vectors-negative300.bin")
      MODE_EXCLUDES+=("--exclude=GoogleNews-vectors-negative300.bin.gz")
    fi
    ;;
esac

echo "Creating remote directory: $REMOTE:$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR"

if [[ "$MODE" == "full" && "$CLEAN_RAW" -eq 1 ]]; then
  echo "Removing stale remote raw JSONL files from $REMOTE:$REMOTE_DIR/data/raw"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "cd $REMOTE_DIR && mkdir -p data/raw && rm -f data/raw/*.jsonl"
fi

echo "Syncing mode '$MODE' to $REMOTE:$REMOTE_DIR"
rsync -ah --partial --info=progress2 \
  -e "$RSYNC_SSH" \
  "${BASE_EXCLUDES[@]}" \
  "${MODE_EXCLUDES[@]}" \
  ./ "$REMOTE:$REMOTE_DIR/"

if [[ "$BOOTSTRAP" -eq 1 ]]; then
  echo "Bootstrapping server virtualenv..."
  ssh "${SSH_OPTS[@]}" "$REMOTE" "cd $REMOTE_DIR && bash scripts/bootstrap_server.sh"
fi

echo "Deploy complete."
