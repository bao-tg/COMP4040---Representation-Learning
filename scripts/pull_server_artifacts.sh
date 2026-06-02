#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-your-server-alias}"
REMOTE_DIR="${REMOTE_DIR:-~/COMP4040---Representation-Learning}"
IDENTITY_FILE="${IDENTITY_FILE:-}"
WITH_EMBEDDINGS=0

usage() {
  cat <<'EOF'
Usage:
  scripts/pull_server_artifacts.sh [--with-embeddings]

Pull report artifacts from a remote server into the current local checkout.

Default pull:
  experiments/server_full/
  logs/
  data/processed/*_summary.json
  data/embeddings/*.meta.json
  data/embeddings/embedding_run_summary.json

Options:
  --with-embeddings  Also pull data/embeddings/*.npy. These files are large.

Environment:
  REMOTE=your-server-alias
  REMOTE_DIR=~/COMP4040---Representation-Learning
  IDENTITY_FILE=/path/to/key  Optional. Omit to use normal SSH config.
EOF
}

SSH_OPTS=()
RSYNC_SSH="ssh"
if [[ -n "$IDENTITY_FILE" ]]; then
  SSH_OPTS=(-o IdentitiesOnly=yes -i "$IDENTITY_FILE")
  RSYNC_SSH="ssh -o IdentitiesOnly=yes -i $IDENTITY_FILE"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-embeddings)
      WITH_EMBEDDINGS=1
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

if [[ -n "$IDENTITY_FILE" && ! -f "$IDENTITY_FILE" ]]; then
  echo "Missing SSH identity: $IDENTITY_FILE" >&2
  exit 1
fi

echo "Checking remote: $REMOTE:$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$REMOTE" "cd $REMOTE_DIR && test -d experiments/server_full"

mkdir -p experiments logs data/processed data/embeddings

echo "Pulling experiment artifacts..."
rsync -ah --partial --info=progress2 -e "$RSYNC_SSH" \
  "$REMOTE:$REMOTE_DIR/experiments/server_full/" \
  "experiments/server_full/"

echo "Pulling logs..."
rsync -ah --partial --info=progress2 -e "$RSYNC_SSH" \
  "$REMOTE:$REMOTE_DIR/logs/" \
  "logs/"

echo "Pulling processed summaries..."
rsync -ah --partial --info=progress2 -e "$RSYNC_SSH" \
  --include='*/' --include='*_summary.json' --exclude='*' \
  "$REMOTE:$REMOTE_DIR/data/processed/" \
  "data/processed/"

echo "Pulling embedding metadata..."
rsync -ah --partial --info=progress2 -e "$RSYNC_SSH" \
  --include='*/' --include='*.meta.json' --include='embedding_run_summary.json' --exclude='*' \
  "$REMOTE:$REMOTE_DIR/data/embeddings/" \
  "data/embeddings/"

if [[ "$WITH_EMBEDDINGS" -eq 1 ]]; then
  echo "Pulling full embedding arrays..."
  rsync -ah --partial --info=progress2 -e "$RSYNC_SSH" \
    --include='*/' --include='*.npy' --exclude='*' \
    "$REMOTE:$REMOTE_DIR/data/embeddings/" \
    "data/embeddings/"
fi

echo "Pull complete."
