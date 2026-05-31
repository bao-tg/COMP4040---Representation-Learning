from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.full_pipeline import (  # noqa: E402
    EXPERIMENTS_DIR,
    RAW_DIR,
    preprocess_raw_sample_to_parquet,
    run_full_anomaly_for_embedding,
    run_full_clustering_for_embeddings,
    run_full_retrieval_for_embeddings,
    sample_embedding_dir,
    sample_processed_path,
    write_all_embeddings,
)


VALID_ENCODERS = ("tfidf", "w2v", "glove", "sbert", "bge")


def parse_encoders(value: str) -> list[str]:
    encoders = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(encoders) - set(VALID_ENCODERS))
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid encoders: {', '.join(invalid)}")
    return encoders


def print_json(title: str, payload: dict) -> None:
    print(f"=== {title} ===", flush=True)
    print(json.dumps(payload, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a resumable COMP4040 sample pipeline on the server.")
    parser.add_argument("--rows-per-category", type=int, default=1_000)
    parser.add_argument("--encoders", type=parse_encoders, default=parse_encoders("tfidf,w2v,sbert,bge"))
    parser.add_argument("--include-glove", action="store_true", help="Include GloVe if glove.840B.300d.txt exists.")
    parser.add_argument(
        "--pretrained-w2v-path",
        type=Path,
        default=PROJECT_ROOT / "GoogleNews-vectors-negative300.bin",
        help="Local Google News Word2Vec binary/gzip path for the w2v encoder.",
    )
    parser.add_argument("--pretrained-w2v-name", default="word2vec-google-news-300")
    parser.add_argument("--download-pretrained-w2v", action="store_true")
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=4096)
    parser.add_argument("--retrieval-queries", type=int, default=1000)
    parser.add_argument("--skip-analysis", action="store_true", help="Only build/reuse embeddings.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild the sample parquet even if it exists.")
    parser.add_argument("--force", action="store_true", help="Rebuild embeddings even if existing files match.")
    args = parser.parse_args()

    encoders = list(args.encoders)
    if args.include_glove and "glove" not in encoders:
        encoders.append("glove")

    processed_path = sample_processed_path(args.rows_per_category)
    if args.force_preprocess or not processed_path.exists():
        if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.jsonl")):
            raise FileNotFoundError(
                f"{processed_path} is missing and no raw JSONL files were found in {RAW_DIR}. "
                "Deploy with mode 'sample' or 'full' first."
            )
        print(f"Creating sample parquet: {processed_path}", flush=True)
        summary = preprocess_raw_sample_to_parquet(rows_per_category=args.rows_per_category)
        print_json("PREPROCESS SAMPLE", summary)
    else:
        print(f"Using existing sample parquet: {processed_path}", flush=True)

    embedding_dir = sample_embedding_dir(args.rows_per_category)
    experiment_dir = EXPERIMENTS_DIR / f"server_sample_{args.rows_per_category}"
    run_glove = "glove" in encoders

    embedding_summary = write_all_embeddings(
        processed_path=processed_path,
        embedding_dir=embedding_dir,
        run_tfidf="tfidf" in encoders,
        run_w2v="w2v" in encoders,
        run_glove=run_glove,
        run_sbert="sbert" in encoders,
        run_bge="bge" in encoders,
        pretrained_w2v_path=args.pretrained_w2v_path,
        pretrained_w2v_name=args.pretrained_w2v_name,
        allow_pretrained_w2v_download=args.download_pretrained_w2v,
        force=args.force,
        transformer_batch_size=args.transformer_batch_size,
        text_batch_size=args.text_batch_size,
    )
    print_json("EMBEDDINGS", embedding_summary)

    available_encoders = [
        encoder for encoder in encoders if (embedding_dir / f"{encoder}.npy").exists()
    ]
    if not available_encoders:
        raise RuntimeError(f"No embeddings were produced in {embedding_dir}")
    if args.skip_analysis:
        print(f"Skipping analysis. Available encoders: {', '.join(available_encoders)}", flush=True)
        return

    clustering = run_full_clustering_for_embeddings(
        processed_path=processed_path,
        embedding_dir=embedding_dir,
        results_dir=experiment_dir / "clustering",
        figures_dir=experiment_dir / "figures",
        encoders=available_encoders,
        label_column="category",
        batch_size=2048,
        umap_plot_sample_size=min(5000, args.rows_per_category * 5),
    )
    print_json("CLUSTERING", clustering)

    retrieval = run_full_retrieval_for_embeddings(
        processed_path=processed_path,
        embedding_dir=embedding_dir,
        results_dir=experiment_dir / "retrieval",
        encoders=available_encoders,
        label_column="category",
        query_sample_size=args.retrieval_queries,
        k_values=[5, 10, 50],
    )
    print_json("RETRIEVAL", retrieval)

    anomaly_encoder = "bge" if "bge" in available_encoders else available_encoders[-1]
    anomaly = run_full_anomaly_for_embedding(
        processed_path=processed_path,
        embedding_path=embedding_dir / f"{anomaly_encoder}.npy",
        results_dir=experiment_dir / "anomaly",
        figures_dir=experiment_dir / "figures",
        contamination=0.01,
        fit_sample_size=args.rows_per_category * 5,
        batch_size=2048,
    )
    print_json("ANOMALY", anomaly)


if __name__ == "__main__":
    main()
