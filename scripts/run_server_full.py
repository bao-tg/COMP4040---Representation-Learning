from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.full_pipeline import (  # noqa: E402
    DEFAULT_GLOVE_PATH,
    DEFAULT_GLOVE_URL,
    DEFAULT_GLOVE_ZIP_PATH,
    EMBEDDING_DIR,
    EXPERIMENTS_DIR,
    PROCESSED_PATH,
    RAW_DIR,
    preprocess_full_raw_to_parquet,
    run_full_anomaly_for_embedding,
    run_full_clustering_for_embeddings,
    run_full_efficiency_summary,
    run_full_linear_probe_for_embeddings,
    run_full_retrieval_for_embeddings,
    run_full_umap_comparison,
    validate_full_analysis_inputs,
    write_final_analysis_summary,
    write_all_embeddings,
    read_json,
    write_json,
)


VALID_ENCODERS = ("tfidf", "w2v", "glove", "sbert", "bge")
VALID_STAGES = ("preprocess", "embeddings", "analysis", "all")
VALID_ANALYSIS_TASKS = (
    "clustering",
    "retrieval",
    "anomaly",
    "umap",
    "linear_probe",
    "efficiency",
    "summary",
    "all",
)


def parse_csv(value: str, valid: tuple[str, ...], label: str) -> list[str]:
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(items) - set(valid))
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid {label}: {', '.join(invalid)}")
    return items


def print_json(title: str, payload: dict) -> None:
    print(f"=== {title} ===", flush=True)
    print(json.dumps(payload, indent=2, default=str), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full COMP4040 server pipeline in resumable stages.")
    parser.add_argument(
        "--stage",
        type=lambda value: parse_csv(value, VALID_STAGES, "stages"),
        default=parse_csv("analysis", VALID_STAGES, "stages"),
        help="Comma-separated stages: preprocess,embeddings,analysis,all",
    )
    parser.add_argument("--encoders", type=lambda value: parse_csv(value, VALID_ENCODERS, "encoders"), default=parse_csv("tfidf,w2v,glove,sbert,bge", VALID_ENCODERS, "encoders"))
    parser.add_argument(
        "--analysis-tasks",
        type=lambda value: parse_csv(value, VALID_ANALYSIS_TASKS, "analysis tasks"),
        default=parse_csv("all", VALID_ANALYSIS_TASKS, "analysis tasks"),
        help="Comma-separated analysis tasks: clustering,retrieval,anomaly,umap,linear_probe,efficiency,summary,all",
    )
    parser.add_argument("--include-glove", action="store_true", help="Include GloVe if glove.840B.300d.txt exists.")
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=DEFAULT_GLOVE_PATH,
        help="Local GloVe 840B 300D text file path for the glove encoder.",
    )
    parser.add_argument(
        "--glove-url",
        default=DEFAULT_GLOVE_URL,
        help="GloVe zip URL used when --download-glove is passed.",
    )
    parser.add_argument(
        "--glove-zip-path",
        type=Path,
        default=DEFAULT_GLOVE_ZIP_PATH,
        help="Local GloVe zip cache path used when --download-glove is passed.",
    )
    parser.add_argument(
        "--download-glove",
        action="store_true",
        help="Download and extract glove.840B.300d.txt if --glove-path is missing.",
    )
    parser.add_argument(
        "--pretrained-w2v-path",
        type=Path,
        default=PROJECT_ROOT / "GoogleNews-vectors-negative300.bin",
        help="Local Google News Word2Vec binary/gzip path for the w2v encoder.",
    )
    parser.add_argument(
        "--pretrained-w2v-name",
        default="word2vec-google-news-300",
        help="gensim-data model name used when --download-pretrained-w2v is passed.",
    )
    parser.add_argument(
        "--download-pretrained-w2v",
        action="store_true",
        help="Allow gensim to download/cache the pretrained Word2Vec model if --pretrained-w2v-path is missing.",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--transformer-batch-size", type=int, default=32)
    parser.add_argument("--text-batch-size", type=int, default=8192)
    parser.add_argument("--retrieval-queries", type=int, default=5000)
    parser.add_argument("--umap-sample-size", type=int, default=50_000)
    parser.add_argument("--force-umap", action="store_true", help="Recompute UMAP coordinates instead of reusing existing coords.")
    parser.add_argument("--linear-probe-batch-size", type=int, default=16_384)
    parser.add_argument("--linear-probe-epochs", type=int, default=3)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--force-embeddings", action="store_true")
    args = parser.parse_args()

    stages = set(args.stage)
    if "all" in stages:
        stages = {"preprocess", "embeddings", "analysis"}

    encoders = list(args.encoders)
    if args.include_glove and "glove" not in encoders:
        encoders.append("glove")

    analysis_tasks = set(args.analysis_tasks)
    if "all" in analysis_tasks:
        analysis_tasks = {"clustering", "retrieval", "anomaly", "umap", "linear_probe", "efficiency", "summary"}

    if "preprocess" in stages:
        if PROCESSED_PATH.exists() and not args.force_preprocess:
            print(f"Using existing processed parquet: {PROCESSED_PATH}", flush=True)
        else:
            if not RAW_DIR.exists() or not list(RAW_DIR.glob("*.jsonl")):
                raise FileNotFoundError(f"No raw JSONL files found in {RAW_DIR}. Deploy with mode 'full' first.")
            summary = preprocess_full_raw_to_parquet(
                chunksize=args.chunksize,
                overwrite=True,
            )
            print_json("PREPROCESS FULL", summary)

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Missing processed parquet: {PROCESSED_PATH}. Run --stage preprocess first.")

    if "embeddings" in stages:
        embedding_summary = write_all_embeddings(
            processed_path=PROCESSED_PATH,
            embedding_dir=EMBEDDING_DIR,
            run_tfidf="tfidf" in encoders,
            run_w2v="w2v" in encoders,
            run_glove="glove" in encoders,
            run_sbert="sbert" in encoders,
            run_bge="bge" in encoders,
            pretrained_w2v_path=args.pretrained_w2v_path,
            pretrained_w2v_name=args.pretrained_w2v_name,
            allow_pretrained_w2v_download=args.download_pretrained_w2v,
            glove_path=args.glove_path,
            glove_url=args.glove_url,
            glove_zip_path=args.glove_zip_path,
            allow_glove_download=args.download_glove,
            force=args.force_embeddings,
            transformer_batch_size=args.transformer_batch_size,
            text_batch_size=args.text_batch_size,
        )
        print_json("EMBEDDINGS FULL", embedding_summary)

    if "analysis" in stages:
        results_root = EXPERIMENTS_DIR / "server_full"
        validation = validate_full_analysis_inputs(
            processed_path=PROCESSED_PATH,
            embedding_dir=EMBEDDING_DIR,
            results_dir=results_root / "validation",
            encoders=encoders,
        )
        print_json("INPUT VALIDATION", validation)
        if not validation["ok"]:
            raise RuntimeError("Input validation failed. See experiments/server_full/validation/input_validation.json")

        if "clustering" in analysis_tasks:
            clustering = run_full_clustering_for_embeddings(
                processed_path=PROCESSED_PATH,
                embedding_dir=EMBEDDING_DIR,
                results_dir=results_root / "clustering",
                figures_dir=results_root / "figures",
                encoders=encoders,
                label_column="category",
                umap_plot_sample_size=0 if "umap" in analysis_tasks else min(args.umap_sample_size, 5_000),
            )
            print_json("CLUSTERING FULL", clustering)

        if "retrieval" in analysis_tasks:
            retrieval = run_full_retrieval_for_embeddings(
                processed_path=PROCESSED_PATH,
                embedding_dir=EMBEDDING_DIR,
                results_dir=results_root / "retrieval",
                encoders=encoders,
                label_column="category",
                query_sample_size=args.retrieval_queries,
                k_values=[5, 10, 50],
            )
            print_json("RETRIEVAL FULL", retrieval)

        if "anomaly" in analysis_tasks:
            all_anomaly_path = results_root / "anomaly" / "all_anomaly_metrics.json"
            anomaly_summaries = read_json(all_anomaly_path) if all_anomaly_path.exists() else {}
            for encoder in encoders:
                anomaly_summaries[encoder.upper()] = run_full_anomaly_for_embedding(
                    processed_path=PROCESSED_PATH,
                    embedding_path=EMBEDDING_DIR / f"{encoder}.npy",
                    results_dir=results_root / "anomaly",
                    figures_dir=results_root / "figures",
                )
            write_json(results_root / "anomaly" / "all_anomaly_metrics.json", anomaly_summaries)
            print_json("ANOMALY FULL", anomaly_summaries)

        if "umap" in analysis_tasks:
            umap_summary = run_full_umap_comparison(
                processed_path=PROCESSED_PATH,
                embedding_dir=EMBEDDING_DIR,
                results_dir=results_root / "umap",
                figures_dir=results_root / "figures",
                clustering_dir=results_root / "clustering",
                encoders=encoders,
                label_column="category",
                sample_size=args.umap_sample_size,
                reuse_existing_coords=not args.force_umap,
            )
            print_json("UMAP FULL", umap_summary)

        if "linear_probe" in analysis_tasks:
            linear_probe = run_full_linear_probe_for_embeddings(
                processed_path=PROCESSED_PATH,
                embedding_dir=EMBEDDING_DIR,
                results_dir=results_root / "linear_probe",
                encoders=encoders,
                target_column="rating",
                batch_size=args.linear_probe_batch_size,
                epochs=args.linear_probe_epochs,
            )
            print_json("LINEAR PROBE FULL", linear_probe)

        if "efficiency" in analysis_tasks:
            efficiency = run_full_efficiency_summary(
                processed_path=PROCESSED_PATH,
                embedding_dir=EMBEDDING_DIR,
                results_dir=results_root / "efficiency",
                encoders=encoders,
            )
            print_json("EFFICIENCY FULL", efficiency)

        if "summary" in analysis_tasks:
            final_summary = write_final_analysis_summary(
                results_root=results_root,
                encoders=encoders,
            )
            print_json("FINAL ANALYSIS SUMMARY", final_summary)


if __name__ == "__main__":
    main()
