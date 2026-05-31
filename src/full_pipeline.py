from __future__ import annotations

import gc
import json
import os
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "cleaned_reviews.parquet"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

RAW_KEEP_COLUMNS = [
    "rating",
    "title",
    "text",
    "user_id",
    "asin",
    "parent_asin",
    "verified_purchase",
    "category",
    "source_file",
    "cleaned_text",
]

RAW_STRING_COLUMNS = [
    "title",
    "text",
    "user_id",
    "asin",
    "parent_asin",
    "category",
    "source_file",
    "cleaned_text",
]

RAW_ARROW_SCHEMA = pa.schema(
    [
        pa.field("rating", pa.float64()),
        pa.field("title", pa.large_string()),
        pa.field("text", pa.large_string()),
        pa.field("user_id", pa.large_string()),
        pa.field("asin", pa.large_string()),
        pa.field("parent_asin", pa.large_string()),
        pa.field("verified_purchase", pa.bool_()),
        pa.field("category", pa.large_string()),
        pa.field("source_file", pa.large_string()),
        pa.field("cleaned_text", pa.large_string()),
    ]
)


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, EMBEDDING_DIR, MODEL_DIR, EXPERIMENTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def category_from_raw_file(path: str | Path) -> str:
    name = Path(path).name
    match = re.match(r"amazon_reviews_(.+?)_sample_\d+\.jsonl$", name)
    if match:
        return match.group(1)
    return Path(path).stem


def discover_raw_files(raw_dir: str | Path = RAW_DIR) -> list[Path]:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("amazon_reviews_*_sample_*.jsonl"))
    if not files:
        files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {raw_dir}")
    return files


def sample_processed_path(rows_per_category: int) -> Path:
    return PROCESSED_DIR / f"cleaned_reviews_sample_{rows_per_category}_per_category.parquet"


def sample_embedding_dir(rows_per_category: int) -> Path:
    return EMBEDDING_DIR / f"sample_{rows_per_category}_per_category"


def clean_text_series(texts: pd.Series, max_tokens: int | None = 256) -> pd.Series:
    cleaned = (
        texts.fillna("")
        .astype(str)
        .str.replace(r"<.*?>", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    if max_tokens is None:
        return cleaned

    def truncate(text: str) -> str:
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    return cleaned.map(truncate)


def _normalize_chunk(chunk: pd.DataFrame, raw_file: Path, max_tokens: int | None) -> pd.DataFrame:
    chunk = chunk.copy()
    for column in RAW_KEEP_COLUMNS:
        if column not in chunk.columns and column not in {"category", "source_file", "cleaned_text"}:
            chunk[column] = pd.NA

    chunk["category"] = category_from_raw_file(raw_file)
    chunk["source_file"] = raw_file.name
    chunk["cleaned_text"] = clean_text_series(chunk["text"], max_tokens=max_tokens)
    chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
    chunk["verified_purchase"] = chunk["verified_purchase"].astype("boolean")
    for column in RAW_STRING_COLUMNS:
        chunk[column] = chunk[column].astype("string")
    return chunk[RAW_KEEP_COLUMNS]


def _raw_table_from_frame(frame: pd.DataFrame) -> pa.Table:
    table = pa.Table.from_pandas(
        frame[RAW_KEEP_COLUMNS],
        schema=RAW_ARROW_SCHEMA,
        preserve_index=False,
        safe=False,
    )
    return table.replace_schema_metadata(None)


def preprocess_full_raw_to_parquet(
    raw_dir: str | Path = RAW_DIR,
    output_path: str | Path = PROCESSED_PATH,
    chunksize: int = 100_000,
    min_length: int = 10,
    max_tokens: int | None = 256,
    deduplicate: bool = True,
    overwrite: bool = True,
) -> dict:
    """
    Stream all raw JSONL files into one cleaned parquet dataset.

    The function avoids loading all 2M raw rows at once. Cross-chunk duplicate
    detection uses stable pandas row hashes over user/product/text keys.
    """
    ensure_dirs()
    raw_files = discover_raw_files(raw_dir)
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use overwrite=True to replace it.")
    if output_path.exists():
        output_path.unlink()

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    writer = None
    seen_hashes: set[int] = set()
    stats = {
        "raw_files": [str(path) for path in raw_files],
        "chunksize": chunksize,
        "min_length": min_length,
        "max_tokens": max_tokens,
        "deduplicate": deduplicate,
        "raw_rows": 0,
        "kept_rows": 0,
        "dropped_empty_or_short": 0,
        "dropped_duplicates": 0,
        "raw_rows_by_category": defaultdict(int),
        "kept_rows_by_category": defaultdict(int),
        "output_path": str(output_path),
    }

    try:
        for raw_file in raw_files:
            category = category_from_raw_file(raw_file)
            reader = pd.read_json(raw_file, lines=True, chunksize=chunksize)
            for chunk in tqdm(reader, desc=f"Preprocess {category}", unit="chunk"):
                stats["raw_rows"] += len(chunk)
                stats["raw_rows_by_category"][category] += len(chunk)

                frame = _normalize_chunk(chunk, raw_file, max_tokens=max_tokens)
                nonempty = frame["cleaned_text"].str.len() >= min_length
                stats["dropped_empty_or_short"] += int((~nonempty).sum())
                frame = frame.loc[nonempty].copy()

                if deduplicate and not frame.empty:
                    dedup_keys = frame[["user_id", "parent_asin", "cleaned_text"]].fillna("")
                    hashes = pd.util.hash_pandas_object(dedup_keys, index=False).astype("uint64")
                    is_new = ~hashes.isin(seen_hashes)
                    stats["dropped_duplicates"] += int((~is_new).sum())
                    seen_hashes.update(int(value) for value in hashes.loc[is_new].to_numpy())
                    frame = frame.loc[is_new].copy()

                if frame.empty:
                    continue

                table = _raw_table_from_frame(frame)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                stats["kept_rows"] += len(frame)
                stats["kept_rows_by_category"][category] += len(frame)

                del chunk, frame, table
                gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if stats["kept_rows"] == 0:
        raise RuntimeError("Preprocessing produced zero rows.")

    stats["raw_rows_by_category"] = dict(stats["raw_rows_by_category"])
    stats["kept_rows_by_category"] = dict(stats["kept_rows_by_category"])
    write_json(summary_path, stats)
    return stats


def preprocess_raw_sample_to_parquet(
    raw_dir: str | Path = RAW_DIR,
    output_path: str | Path | None = None,
    rows_per_category: int = 1_000,
    chunksize: int = 25_000,
    min_length: int = 10,
    max_tokens: int | None = 256,
    deduplicate: bool = True,
    overwrite: bool = True,
) -> dict:
    """
    Build a small local parquet artifact from every raw category file.

    Raw files in this project are already sampled JSONL exports, so taking the
    first cleaned rows from each category is enough for a fast local smoke run.
    The full server run should use preprocess_full_raw_to_parquet instead.
    """
    ensure_dirs()
    raw_files = discover_raw_files(raw_dir)
    output_path = Path(output_path) if output_path else sample_processed_path(rows_per_category)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use overwrite=True to replace it.")
    if output_path.exists():
        output_path.unlink()

    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    writer = None
    seen_hashes: set[int] = set()
    stats = {
        "mode": "sample",
        "raw_files": [str(path) for path in raw_files],
        "rows_per_category": rows_per_category,
        "chunksize": chunksize,
        "min_length": min_length,
        "max_tokens": max_tokens,
        "deduplicate": deduplicate,
        "raw_rows_scanned": 0,
        "kept_rows": 0,
        "dropped_empty_or_short": 0,
        "dropped_duplicates": 0,
        "kept_rows_by_category": defaultdict(int),
        "output_path": str(output_path),
    }

    try:
        for raw_file in raw_files:
            category = category_from_raw_file(raw_file)
            reader = pd.read_json(raw_file, lines=True, chunksize=chunksize)
            for chunk in tqdm(reader, desc=f"Sample {category}", unit="chunk"):
                if stats["kept_rows_by_category"][category] >= rows_per_category:
                    break

                stats["raw_rows_scanned"] += len(chunk)
                frame = _normalize_chunk(chunk, raw_file, max_tokens=max_tokens)
                nonempty = frame["cleaned_text"].str.len() >= min_length
                stats["dropped_empty_or_short"] += int((~nonempty).sum())
                frame = frame.loc[nonempty].copy()

                if deduplicate and not frame.empty:
                    dedup_keys = frame[["user_id", "parent_asin", "cleaned_text"]].fillna("")
                    hashes = pd.util.hash_pandas_object(dedup_keys, index=False).astype("uint64")
                    is_new = ~hashes.isin(seen_hashes)
                    stats["dropped_duplicates"] += int((~is_new).sum())
                    seen_hashes.update(int(value) for value in hashes.loc[is_new].to_numpy())
                    frame = frame.loc[is_new].copy()

                remaining = rows_per_category - stats["kept_rows_by_category"][category]
                frame = frame.head(remaining)
                if frame.empty:
                    continue

                table = _raw_table_from_frame(frame)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table)
                stats["kept_rows"] += len(frame)
                stats["kept_rows_by_category"][category] += len(frame)

                del chunk, frame, table
                gc.collect()
    finally:
        if writer is not None:
            writer.close()

    if stats["kept_rows"] == 0:
        raise RuntimeError("Sample preprocessing produced zero rows.")

    stats["kept_rows_by_category"] = dict(stats["kept_rows_by_category"])
    write_json(summary_path, stats)
    return stats


def summarize_raw_jsonl(raw_dir: str | Path = RAW_DIR, chunksize: int = 100_000) -> dict:
    raw_files = discover_raw_files(raw_dir)
    rating_counts = Counter()
    category_counts = Counter()
    verified_true = 0
    verified_total = 0
    text_lengths: list[int] = []
    word_counts: list[int] = []

    for raw_file in raw_files:
        category = category_from_raw_file(raw_file)
        reader = pd.read_json(raw_file, lines=True, chunksize=chunksize)
        for chunk in tqdm(reader, desc=f"EDA {category}", unit="chunk"):
            category_counts[category] += len(chunk)
            rating_counts.update(chunk["rating"].dropna().astype(str).tolist())
            if "verified_purchase" in chunk:
                verified = chunk["verified_purchase"].dropna().astype(bool)
                verified_true += int(verified.sum())
                verified_total += int(len(verified))
            texts = chunk["text"].fillna("").astype(str)
            text_lengths.extend(texts.str.len().astype(int).tolist())
            word_counts.extend(texts.str.split().str.len().astype(int).tolist())

    length_series = pd.Series(text_lengths, dtype="int64")
    word_series = pd.Series(word_counts, dtype="int64")
    return {
        "raw_files": [str(path) for path in raw_files],
        "total_rows": int(sum(category_counts.values())),
        "category_counts": dict(category_counts),
        "rating_counts": dict(sorted(rating_counts.items(), key=lambda item: float(item[0]))),
        "verified_purchase_percent": (
            (verified_true / verified_total) * 100 if verified_total else None
        ),
        "text_length_describe": length_series.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
        "word_count_describe": word_series.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).to_dict(),
    }


def parquet_row_count(path: str | Path = PROCESSED_PATH) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def parquet_columns(path: str | Path = PROCESSED_PATH) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def load_label_array(path: str | Path = PROCESSED_PATH, preferred: str = "category") -> tuple[np.ndarray, str]:
    columns = parquet_columns(path)
    label_col = preferred if preferred in columns else "rating"
    labels = pd.read_parquet(path, columns=[label_col])[label_col].to_numpy()
    return labels, label_col


def iter_text_batches(
    path: str | Path = PROCESSED_PATH,
    batch_size: int = 8192,
    column: str = "cleaned_text",
):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=[column]):
        values = batch.column(0).to_pylist()
        yield [value if isinstance(value, str) else "" for value in values]


def iter_texts(path: str | Path = PROCESSED_PATH, batch_size: int = 8192):
    for batch in iter_text_batches(path, batch_size=batch_size):
        yield from batch


class ParquetTokenIterable:
    def __init__(self, path: str | Path = PROCESSED_PATH, batch_size: int = 8192):
        self.path = Path(path)
        self.batch_size = batch_size

    def __iter__(self):
        for batch in iter_text_batches(self.path, batch_size=self.batch_size):
            for text in batch:
                yield text.split()


def _open_output_memmap(output_path: Path, shape: tuple[int, int], resume: bool = True):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and resume:
        existing = np.load(output_path, mmap_mode="r+")
        if existing.shape != shape:
            print(f"{output_path} has shape {existing.shape}, expected {shape}; rebuilding it.")
            del existing
            output_path.unlink()
            meta_path = _embedding_meta_path(output_path)
            if meta_path.exists():
                meta_path.unlink()
            return np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=shape)
        return existing
    return np.lib.format.open_memmap(output_path, mode="w+", dtype="float32", shape=shape)


def _reuse_existing_embedding(output_path: Path, expected_shape: tuple[int, int], force: bool) -> dict | None:
    if force or not output_path.exists():
        return None
    arr = np.load(output_path, mmap_mode="r")
    if arr.shape == expected_shape:
        return {"path": str(output_path), "shape": list(arr.shape), "skipped_existing": True}
    print(f"{output_path} has shape {arr.shape}, expected {expected_shape}; rebuilding it.")
    del arr
    output_path.unlink()
    meta_path = _embedding_meta_path(output_path)
    if meta_path.exists():
        meta_path.unlink()
    return None


def _reuse_existing_embedding_with_max_dim(
    output_path: Path,
    expected_rows: int,
    max_dim: int,
    force: bool,
) -> dict | None:
    if force or not output_path.exists():
        return None
    arr = np.load(output_path, mmap_mode="r")
    if arr.shape[0] == expected_rows and 1 <= arr.shape[1] <= max_dim:
        return {"path": str(output_path), "shape": list(arr.shape), "skipped_existing": True}
    print(f"{output_path} has shape {arr.shape}, expected rows {expected_rows} and dim <= {max_dim}; rebuilding it.")
    del arr
    output_path.unlink()
    meta_path = _embedding_meta_path(output_path)
    if meta_path.exists():
        meta_path.unlink()
    return None


def _embedding_meta_path(output_path: str | Path) -> Path:
    return Path(str(output_path) + ".meta.json")


def _read_rows_written(meta_path: Path, expected_rows: int, expected_dim: int) -> int:
    if not meta_path.exists():
        return 0
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("total_rows") != expected_rows or meta.get("dim") != expected_dim:
        return 0
    return int(meta.get("rows_written", 0))


def _write_embedding_meta(output_path: Path, rows_written: int, total_rows: int, dim: int, kind: str) -> None:
    write_json(
        _embedding_meta_path(output_path),
        {
            "kind": kind,
            "path": str(output_path),
            "rows_written": rows_written,
            "total_rows": total_rows,
            "dim": dim,
            "complete": rows_written >= total_rows,
        },
    )


def write_transformer_embeddings(
    processed_path: str | Path,
    output_path: str | Path,
    model_name: str,
    batch_size: int = 256,
    text_batch_size: int | None = None,
    resume: bool = True,
) -> dict:
    from sentence_transformers import SentenceTransformer
    import torch

    processed_path = Path(processed_path)
    output_path = Path(output_path)
    total_rows = parquet_row_count(processed_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    dim = int(model.get_sentence_embedding_dimension())

    output = _open_output_memmap(output_path, (total_rows, dim), resume=resume)
    meta_path = _embedding_meta_path(output_path)
    rows_written = _read_rows_written(meta_path, total_rows, dim) if output_path.exists() else 0
    text_batch_size = text_batch_size or batch_size

    offset = 0
    progress = tqdm(total=total_rows, initial=rows_written, desc=f"Encode {output_path.stem}")
    for texts in iter_text_batches(processed_path, batch_size=text_batch_size):
        batch_len = len(texts)
        batch_start = offset
        batch_end = offset + batch_len
        offset = batch_end
        if batch_end <= rows_written:
            continue
        if batch_start < rows_written:
            texts = texts[rows_written - batch_start :]
            batch_start = rows_written

        vectors = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32", copy=False)
        output[batch_start : batch_start + len(vectors)] = vectors
        output.flush()
        rows_written = batch_start + len(vectors)
        _write_embedding_meta(output_path, rows_written, total_rows, dim, model_name)
        progress.update(len(vectors))
    progress.close()

    _write_embedding_meta(output_path, total_rows, total_rows, dim, model_name)
    return {"path": str(output_path), "shape": [total_rows, dim], "device": device}


def write_tfidf_svd_embeddings(
    processed_path: str | Path = PROCESSED_PATH,
    output_path: str | Path = EMBEDDING_DIR / "tfidf.npy",
    n_components: int = 300,
    max_features: int = 10_000,
    text_batch_size: int = 8192,
    random_state: int = 42,
    force: bool = False,
) -> dict:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    processed_path = Path(processed_path)
    output_path = Path(output_path)
    model_path = MODEL_DIR / "tfidf_svd.pkl"
    total_rows = parquet_row_count(processed_path)
    reusable = _reuse_existing_embedding_with_max_dim(output_path, total_rows, n_components, force=force)
    if reusable is not None:
        return reusable

    print("Fitting TF-IDF vocabulary and sparse matrix on full cleaned corpus...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(iter_texts(processed_path, batch_size=text_batch_size))
    effective_components = min(n_components, tfidf_matrix.shape[0], tfidf_matrix.shape[1])
    if effective_components < 1:
        raise RuntimeError("TF-IDF produced zero features.")
    if effective_components < n_components:
        print(
            f"TF-IDF matrix shape is {tfidf_matrix.shape}; "
            f"using {effective_components} SVD components instead of {n_components}."
        )

    print("Fitting TruncatedSVD...")
    svd = TruncatedSVD(n_components=effective_components, random_state=random_state)
    svd.fit(tfidf_matrix)
    del tfidf_matrix
    gc.collect()

    output = _open_output_memmap(output_path, (total_rows, effective_components), resume=False)
    offset = 0
    for texts in tqdm(iter_text_batches(processed_path, batch_size=text_batch_size), desc="Transform TF-IDF"):
        transformed = svd.transform(vectorizer.transform(texts)).astype("float32", copy=False)
        output[offset : offset + len(texts)] = transformed
        offset += len(texts)
    output.flush()

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "svd": svd}, handle)
    _write_embedding_meta(output_path, total_rows, total_rows, effective_components, "tfidf_svd")
    return {"path": str(output_path), "shape": [total_rows, effective_components], "model_path": str(model_path)}


def _mean_embedding(tokens: list[str], keyed_vectors, vector_size: int) -> np.ndarray:
    vectors = [keyed_vectors[word] for word in tokens if word in keyed_vectors]
    if not vectors:
        return np.zeros(vector_size, dtype="float32")
    return np.mean(vectors, axis=0).astype("float32", copy=False)


def write_pretrained_w2v_embeddings(
    processed_path: str | Path = PROCESSED_PATH,
    output_path: str | Path = EMBEDDING_DIR / "w2v.npy",
    pretrained_path: str | Path | None = PROJECT_ROOT / "GoogleNews-vectors-negative300.bin",
    pretrained_name: str = "word2vec-google-news-300",
    allow_download: bool = False,
    vector_size: int = 300,
    text_batch_size: int = 8192,
    force: bool = False,
) -> dict:
    processed_path = Path(processed_path)
    output_path = Path(output_path)
    total_rows = parquet_row_count(processed_path)
    reusable = _reuse_existing_embedding(output_path, (total_rows, vector_size), force=force)
    if reusable is not None:
        return reusable

    source = None
    keyed_vectors = None
    pretrained_path_obj = Path(pretrained_path) if pretrained_path else None
    if pretrained_path_obj and not pretrained_path_obj.exists() and pretrained_path_obj.suffix != ".gz":
        gz_fallback = pretrained_path_obj.with_suffix(pretrained_path_obj.suffix + ".gz")
        if gz_fallback.exists():
            pretrained_path_obj = gz_fallback
    if pretrained_path_obj and pretrained_path_obj.exists():
        from gensim.models import KeyedVectors

        source = str(pretrained_path_obj)
        print(f"Loading pretrained Word2Vec vectors from {source}...")
        keyed_vectors = KeyedVectors.load_word2vec_format(str(pretrained_path_obj), binary=True)
    elif allow_download:
        import gensim.downloader as api

        source = f"gensim-data:{pretrained_name}"
        print(f"Loading pretrained Word2Vec vectors from {source}...")
        keyed_vectors = api.load(pretrained_name)
    else:
        return {
            "path": str(output_path),
            "skipped_missing_pretrained_w2v": True,
            "pretrained_path": str(pretrained_path_obj) if pretrained_path_obj else None,
            "pretrained_name": pretrained_name,
            "download_hint": "Pass --download-pretrained-w2v or place GoogleNews-vectors-negative300.bin(.gz) at the repo root.",
        }

    actual_vector_size = int(keyed_vectors.vector_size)
    if actual_vector_size != vector_size:
        raise ValueError(
            f"Pretrained Word2Vec vector size is {actual_vector_size}, expected {vector_size}. "
            "Use a 300d Word2Vec source for this pipeline."
        )

    vocabulary = collect_vocabulary(processed_path, batch_size=text_batch_size)
    covered_words = sum(1 for word in vocabulary if word in keyed_vectors)
    print(f"Covered {covered_words:,}/{len(vocabulary):,} corpus words with pretrained Word2Vec.")

    output = _open_output_memmap(output_path, (total_rows, vector_size), resume=False)
    offset = 0
    for texts in tqdm(iter_text_batches(processed_path, batch_size=text_batch_size), desc="Encode pretrained W2V"):
        rows = [_mean_embedding(text.split(), keyed_vectors, vector_size) for text in texts]
        vectors = np.vstack(rows).astype("float32", copy=False)
        output[offset : offset + len(vectors)] = vectors
        offset += len(vectors)
    output.flush()
    _write_embedding_meta(output_path, total_rows, total_rows, vector_size, f"pretrained_word2vec:{source}")
    return {
        "path": str(output_path),
        "shape": [total_rows, vector_size],
        "pretrained_source": source,
        "pretrained_name": pretrained_name,
        "covered_words": int(covered_words),
        "vocabulary_size": int(len(vocabulary)),
    }


def collect_vocabulary(
    processed_path: str | Path = PROCESSED_PATH,
    batch_size: int = 8192,
) -> set[str]:
    vocabulary: set[str] = set()
    for texts in tqdm(iter_text_batches(processed_path, batch_size=batch_size), desc="Collect vocabulary"):
        for text in texts:
            vocabulary.update(text.split())
    return vocabulary


def load_glove_subset(
    glove_path: str | Path,
    vector_size: int = 300,
    vocabulary: set[str] | None = None,
) -> dict[str, np.ndarray]:
    glove_path = Path(glove_path)
    embeddings: dict[str, np.ndarray] = {}
    with glove_path.open("r", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Load GloVe"):
            values = line.rstrip().split(" ")
            if len(values) < vector_size + 1:
                continue
            word = " ".join(values[:-vector_size])
            if vocabulary is not None and word not in vocabulary:
                continue
            coefs = np.asarray(values[-vector_size:], dtype="float32")
            embeddings[word] = coefs
    return embeddings


def write_glove_embeddings(
    processed_path: str | Path = PROCESSED_PATH,
    glove_path: str | Path = PROJECT_ROOT / "glove.840B.300d.txt",
    output_path: str | Path = EMBEDDING_DIR / "glove.npy",
    vector_size: int = 300,
    text_batch_size: int = 8192,
    force: bool = False,
) -> dict:
    processed_path = Path(processed_path)
    glove_path = Path(glove_path)
    output_path = Path(output_path)
    total_rows = parquet_row_count(processed_path)
    reusable = _reuse_existing_embedding(output_path, (total_rows, vector_size), force=force)
    if reusable is not None:
        return reusable
    if not glove_path.exists():
        return {"path": str(output_path), "skipped_missing_glove": True, "glove_path": str(glove_path)}

    vocabulary = collect_vocabulary(processed_path, batch_size=text_batch_size)
    embeddings = load_glove_subset(glove_path, vector_size=vector_size, vocabulary=vocabulary)
    print(f"Loaded {len(embeddings):,}/{len(vocabulary):,} corpus words from GloVe.")
    output = _open_output_memmap(output_path, (total_rows, vector_size), resume=False)
    offset = 0
    for texts in tqdm(iter_text_batches(processed_path, batch_size=text_batch_size), desc="Encode GloVe"):
        rows = [_mean_embedding(text.split(), embeddings, vector_size) for text in texts]
        vectors = np.vstack(rows).astype("float32", copy=False)
        output[offset : offset + len(vectors)] = vectors
        offset += len(vectors)
    output.flush()
    _write_embedding_meta(output_path, total_rows, total_rows, vector_size, "glove")
    return {"path": str(output_path), "shape": [total_rows, vector_size], "glove_path": str(glove_path)}


def write_all_embeddings(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_dir: str | Path = EMBEDDING_DIR,
    run_tfidf: bool = True,
    run_w2v: bool = True,
    run_glove: bool = True,
    run_sbert: bool = True,
    run_bge: bool = True,
    bge_model_name: str = "BAAI/bge-large-en-v1.5",
    pretrained_w2v_path: str | Path | None = PROJECT_ROOT / "GoogleNews-vectors-negative300.bin",
    pretrained_w2v_name: str = "word2vec-google-news-300",
    allow_pretrained_w2v_download: bool = False,
    force: bool = False,
    transformer_batch_size: int = 256,
    text_batch_size: int = 8192,
) -> dict:
    embedding_dir = Path(embedding_dir)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    if run_tfidf:
        results["tfidf"] = write_tfidf_svd_embeddings(
            processed_path=processed_path,
            output_path=embedding_dir / "tfidf.npy",
            text_batch_size=text_batch_size,
            force=force,
        )
    if run_w2v:
        results["w2v"] = write_pretrained_w2v_embeddings(
            processed_path=processed_path,
            output_path=embedding_dir / "w2v.npy",
            pretrained_path=pretrained_w2v_path,
            pretrained_name=pretrained_w2v_name,
            allow_download=allow_pretrained_w2v_download,
            text_batch_size=text_batch_size,
            force=force,
        )
    if run_glove:
        results["glove"] = write_glove_embeddings(
            processed_path=processed_path,
            output_path=embedding_dir / "glove.npy",
            text_batch_size=text_batch_size,
            force=force,
        )
    if run_sbert:
        results["sbert"] = write_transformer_embeddings(
            processed_path=processed_path,
            output_path=embedding_dir / "sbert.npy",
            model_name="all-MiniLM-L6-v2",
            batch_size=transformer_batch_size,
            text_batch_size=transformer_batch_size,
            resume=not force,
        )
    if run_bge:
        results["bge"] = write_transformer_embeddings(
            processed_path=processed_path,
            output_path=embedding_dir / "bge.npy",
            model_name=bge_model_name,
            batch_size=transformer_batch_size,
            text_batch_size=transformer_batch_size,
            resume=not force,
        )
    summary_path = embedding_dir / "embedding_run_summary.json"
    existing_summary = read_json(summary_path) if summary_path.exists() else {}
    existing_summary.update(results)
    write_json(summary_path, existing_summary)
    return results


def load_embedding(path: str | Path):
    return np.load(path, mmap_mode="r")


def _batch_slices(n_rows: int, batch_size: int):
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


def _human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024 or unit == "TB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{num_bytes} B"


def sampled_metric_frame(X, labels_true, labels_pred, sample_size: int = 20_000, random_state: int = 42) -> dict:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

    metrics = {
        "nmi": float(normalized_mutual_info_score(labels_true, labels_pred)),
        "ari": float(adjusted_rand_score(labels_true, labels_pred)),
    }
    rng = np.random.default_rng(random_state)
    n_rows = len(labels_pred)
    sample_size = min(sample_size, n_rows)
    indices = rng.choice(n_rows, size=sample_size, replace=False)
    unique_pred = np.unique(labels_pred[indices])
    if 1 < len(unique_pred) < sample_size:
        metrics["silhouette"] = float(silhouette_score(X[indices], labels_pred[indices]))
        metrics["silhouette_sample_size"] = int(sample_size)
    else:
        metrics["silhouette"] = float("nan")
        metrics["silhouette_sample_size"] = int(sample_size)
    return metrics


def run_minibatch_kmeans_full(
    X,
    labels_true: np.ndarray,
    n_clusters: int,
    batch_size: int = 16_384,
    passes: int = 1,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    from sklearn.cluster import MiniBatchKMeans

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
    )
    n_rows = X.shape[0]
    for epoch in range(passes):
        for start, end in tqdm(list(_batch_slices(n_rows, batch_size)), desc=f"KMeans fit pass {epoch + 1}"):
            kmeans.partial_fit(np.asarray(X[start:end], dtype="float32", order="C"))

    labels_pred = np.empty(n_rows, dtype="int32")
    for start, end in tqdm(list(_batch_slices(n_rows, batch_size)), desc="KMeans predict"):
        labels_pred[start:end] = kmeans.predict(np.asarray(X[start:end], dtype="float32", order="C"))

    metrics = sampled_metric_frame(X, labels_true, labels_pred, random_state=random_state)
    return labels_pred, metrics


def save_umap_sample_plot(
    X,
    labels,
    output_path: str | Path,
    title: str,
    sample_size: int = 5_000,
    random_state: int = 42,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import umap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_state)
    n_rows = X.shape[0]
    sample_size = min(sample_size, n_rows)
    indices = rng.choice(n_rows, size=sample_size, replace=False)
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    coords = reducer.fit_transform(np.asarray(X[indices], dtype="float32", order="C"))

    plt.figure(figsize=(9, 7))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=np.asarray(labels)[indices], s=8, linewidth=0)
    plt.title(title)
    plt.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_full_clustering_for_embeddings(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_dir: str | Path = EMBEDDING_DIR,
    results_dir: str | Path = EXPERIMENTS_DIR / "clustering",
    figures_dir: str | Path = EXPERIMENTS_DIR / "figures",
    encoders: list[str] | None = None,
    label_column: str = "category",
    batch_size: int = 16_384,
    umap_plot_sample_size: int | None = 5_000,
) -> dict:
    embedding_dir = Path(embedding_dir)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    encoders = encoders or ["tfidf", "w2v", "glove", "sbert", "bge"]
    labels_true, actual_label_column = load_label_array(processed_path, preferred=label_column)
    n_clusters = int(len(np.unique(labels_true)))
    all_metrics = {}

    for encoder in encoders:
        emb_path = embedding_dir / f"{encoder}.npy"
        if not emb_path.exists():
            print(f"Skipping {encoder}: {emb_path} not found")
            continue
        X = load_embedding(emb_path)
        if X.shape[0] != len(labels_true):
            raise ValueError(f"{encoder} rows {X.shape[0]} do not match labels {len(labels_true)}")
        labels_pred, metrics = run_minibatch_kmeans_full(
            X,
            labels_true,
            n_clusters=n_clusters,
            batch_size=batch_size,
        )
        metrics["label_column"] = actual_label_column
        metrics["n_clusters"] = n_clusters
        all_metrics[encoder.upper()] = metrics
        np.save(results_dir / f"{encoder}_cluster_assignments.npy", labels_pred)
        write_json(results_dir / f"{encoder}_metrics.json", metrics)
        if umap_plot_sample_size and umap_plot_sample_size > 0:
            save_umap_sample_plot(
                X,
                labels_true,
                figures_dir / f"{encoder}_umap_gt.png",
                f"{encoder.upper()} Embeddings ({actual_label_column})",
                sample_size=umap_plot_sample_size,
            )
            save_umap_sample_plot(
                X,
                labels_pred,
                figures_dir / f"{encoder}_umap_clusters.png",
                f"{encoder.upper()} MiniBatchKMeans Clusters",
                sample_size=umap_plot_sample_size,
            )
        del X, labels_pred
        gc.collect()

    all_metrics_path = results_dir / "all_clustering_metrics.json"
    merged_metrics = read_json(all_metrics_path) if all_metrics_path.exists() else {}
    merged_metrics.update(all_metrics)
    write_json(all_metrics_path, merged_metrics)
    legacy_path = results_dir / "all_metrics.json"
    if legacy_path.exists():
        legacy_path.unlink()
    return merged_metrics


def evaluate_retrieval_full(
    X,
    labels: np.ndarray,
    sample_size: int = 5_000,
    k_values: list[int] | None = None,
    batch_size: int = 65_536,
    random_state: int = 42,
) -> dict:
    import faiss

    k_values = k_values or [5, 10, 50]
    n_rows, dim = X.shape
    index = faiss.IndexFlatL2(dim)
    for start, end in tqdm(list(_batch_slices(n_rows, batch_size)), desc="Build FAISS index"):
        index.add(np.asarray(X[start:end], dtype="float32", order="C"))

    rng = np.random.default_rng(random_state)
    n_queries = min(sample_size, n_rows)
    query_indices = rng.choice(n_rows, size=n_queries, replace=False)
    queries = np.asarray(X[query_indices], dtype="float32", order="C")
    query_labels = np.asarray(labels)[query_indices]

    max_k = max(k_values)
    distances, indices = index.search(queries, max_k + 1)
    del distances

    results = {"mrr": 0.0}
    for k in k_values:
        results[f"precision@{k}"] = 0.0

    all_labels = np.asarray(labels)
    for i in tqdm(range(n_queries), desc="Evaluate retrieval"):
        pred_indices = indices[i]
        if pred_indices[0] == query_indices[i]:
            pred_indices = pred_indices[1:]
        else:
            pred_indices = pred_indices[:max_k]
        pred_labels = all_labels[pred_indices]
        actual = query_labels[i]
        for k in k_values:
            results[f"precision@{k}"] += float(np.mean(pred_labels[:k] == actual))
        matches = np.flatnonzero(pred_labels == actual)
        results["mrr"] += float(1.0 / (matches[0] + 1)) if len(matches) else 0.0

    for key in results:
        results[key] /= n_queries
    results["query_sample_size"] = int(n_queries)
    return results


def run_full_retrieval_for_embeddings(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_dir: str | Path = EMBEDDING_DIR,
    results_dir: str | Path = EXPERIMENTS_DIR / "retrieval",
    encoders: list[str] | None = None,
    label_column: str = "category",
    query_sample_size: int = 5_000,
    k_values: list[int] | None = None,
) -> dict:
    embedding_dir = Path(embedding_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    encoders = encoders or ["tfidf", "w2v", "glove", "sbert", "bge"]
    labels_true, actual_label_column = load_label_array(processed_path, preferred=label_column)
    all_metrics = {}

    for encoder in encoders:
        emb_path = embedding_dir / f"{encoder}.npy"
        if not emb_path.exists():
            print(f"Skipping {encoder}: {emb_path} not found")
            continue
        X = load_embedding(emb_path)
        if X.shape[0] != len(labels_true):
            raise ValueError(f"{encoder} rows {X.shape[0]} do not match labels {len(labels_true)}")
        metrics = evaluate_retrieval_full(
            X,
            labels_true,
            sample_size=query_sample_size,
            k_values=k_values,
        )
        metrics["label_column"] = actual_label_column
        all_metrics[encoder.upper()] = metrics
        write_json(results_dir / f"{encoder}_retrieval_metrics.json", metrics)
        del X
        gc.collect()

    all_metrics_path = results_dir / "all_retrieval_metrics.json"
    merged_metrics = read_json(all_metrics_path) if all_metrics_path.exists() else {}
    merged_metrics.update(all_metrics)
    write_json(all_metrics_path, merged_metrics)
    return merged_metrics


def run_full_anomaly_for_embedding(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_path: str | Path = EMBEDDING_DIR / "bge.npy",
    results_dir: str | Path = EXPERIMENTS_DIR / "anomaly",
    figures_dir: str | Path = EXPERIMENTS_DIR / "figures",
    contamination: float = 0.01,
    fit_sample_size: int = 200_000,
    batch_size: int = 16_384,
    random_state: int = 42,
) -> dict:
    from sklearn.ensemble import IsolationForest
    from sklearn.linear_model import SGDRegressor
    import matplotlib.pyplot as plt
    import seaborn as sns

    processed_path = Path(processed_path)
    embedding_path = Path(embedding_path)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X = load_embedding(embedding_path)
    meta = pd.read_parquet(
        processed_path,
        columns=["rating", "category", "title", "cleaned_text"],
    )
    ratings = meta["rating"].astype("float32").to_numpy()
    n_rows = X.shape[0]
    if len(ratings) != n_rows:
        raise ValueError(f"Embedding rows {n_rows} do not match processed rows {len(ratings)}")

    rng = np.random.default_rng(random_state)
    sample_size = min(fit_sample_size, n_rows)
    sample_indices = rng.choice(n_rows, size=sample_size, replace=False)
    iso = IsolationForest(
        contamination=contamination,
        max_samples=min(10_000, sample_size),
        random_state=random_state,
        n_jobs=-1,
    )
    print(f"Fitting IsolationForest on {sample_size} sampled rows, then scoring all {n_rows} rows...")
    iso.fit(np.asarray(X[sample_indices], dtype="float32", order="C"))
    iso_scores = np.empty(n_rows, dtype="float32")
    for start, end in tqdm(list(_batch_slices(n_rows, batch_size)), desc="Score IsolationForest"):
        iso_scores[start:end] = iso.decision_function(np.asarray(X[start:end], dtype="float32", order="C"))
    iso_threshold = float(np.quantile(iso_scores, contamination))
    is_iso_anomaly = iso_scores <= iso_threshold

    print("Training streaming SGDRegressor for rating inconsistency...")
    reg = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=random_state,
    )
    for epoch in range(3):
        for start, end in tqdm(list(_batch_slices(n_rows, batch_size)), desc=f"Rating model epoch {epoch + 1}"):
            reg.partial_fit(np.asarray(X[start:end], dtype="float32", order="C"), ratings[start:end])

    predicted = np.empty(n_rows, dtype="float32")
    for start, end in tqdm(list(_batch_slices(n_rows, batch_size)), desc="Predict ratings"):
        predicted[start:end] = reg.predict(np.asarray(X[start:end], dtype="float32", order="C"))
    residuals = np.abs(ratings - predicted)
    n_flag = max(1, int(n_rows * contamination))
    residual_threshold = float(np.partition(residuals, -n_flag)[-n_flag])
    is_inconsistent = residuals >= residual_threshold

    meta = meta.copy()
    meta["iso_anomaly"] = is_iso_anomaly
    meta["iso_score"] = iso_scores
    meta["inconsistent"] = is_inconsistent
    meta["rating_residual"] = residuals
    meta["predicted_rating"] = predicted

    iso_csv = results_dir / f"{embedding_path.stem}_iso_anomalies.csv"
    inc_csv = results_dir / f"{embedding_path.stem}_inconsistencies.csv"
    meta.loc[is_iso_anomaly].sort_values("iso_score").to_csv(iso_csv, index=False)
    meta.loc[is_inconsistent].sort_values("rating_residual", ascending=False).to_csv(inc_csv, index=False)

    plt.figure(figsize=(8, 4))
    sns.histplot(iso_scores, bins=80, kde=True)
    plt.title("Isolation Forest Anomaly Scores")
    plt.tight_layout()
    iso_plot = figures_dir / f"{embedding_path.stem}_iso_forest_scores.png"
    residual_plot = figures_dir / f"{embedding_path.stem}_rating_inconsistency_residuals.png"

    plt.savefig(iso_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.histplot(residuals, bins=80, kde=True)
    plt.title("Rating Prediction Absolute Residuals")
    plt.tight_layout()
    plt.savefig(residual_plot, dpi=160)
    plt.close()

    summary = {
        "embedding_path": str(embedding_path),
        "rows": int(n_rows),
        "contamination": contamination,
        "isolation_fit_sample_size": int(sample_size),
        "outlier_model": "IsolationForest",
        "rating_inconsistency_model": "SGDRegressor(loss=squared_error, penalty=l2)",
        "rating_inconsistency_epochs": 3,
        "iso_anomaly_count": int(is_iso_anomaly.sum()),
        "iso_score_threshold": iso_threshold,
        "inconsistency_count": int(is_inconsistent.sum()),
        "rating_residual_threshold": residual_threshold,
        "iso_csv": str(iso_csv),
        "inconsistency_csv": str(inc_csv),
        "iso_plot": str(iso_plot),
        "residual_plot": str(residual_plot),
    }
    write_json(results_dir / f"{embedding_path.stem}_anomaly_summary.json", summary)
    return summary


def validate_full_analysis_inputs(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_dir: str | Path = EMBEDDING_DIR,
    results_dir: str | Path = EXPERIMENTS_DIR / "server_full" / "validation",
    encoders: list[str] | None = None,
    required_columns: list[str] | None = None,
) -> dict:
    processed_path = Path(processed_path)
    embedding_dir = Path(embedding_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    encoders = encoders or ["tfidf", "w2v", "glove", "sbert", "bge"]
    required_columns = required_columns or ["rating", "category", "cleaned_text"]

    errors: list[str] = []
    if not processed_path.exists():
        errors.append(f"Missing processed parquet: {processed_path}")
        summary = {"ok": False, "errors": errors, "encoders": {}}
        write_json(results_dir / "input_validation.json", summary)
        return summary

    processed_rows = parquet_row_count(processed_path)
    columns = parquet_columns(processed_path)
    missing_columns = [column for column in required_columns if column not in columns]
    if missing_columns:
        errors.append(f"Missing processed columns: {', '.join(missing_columns)}")

    encoder_results: dict[str, dict] = {}
    for encoder in encoders:
        key = encoder.upper()
        emb_path = embedding_dir / f"{encoder}.npy"
        meta_path = _embedding_meta_path(emb_path)
        encoder_errors: list[str] = []
        shape = None
        file_size = 0
        meta = {}

        if not emb_path.exists():
            encoder_errors.append(f"Missing embedding: {emb_path}")
        else:
            arr = np.load(emb_path, mmap_mode="r")
            shape = list(arr.shape)
            del arr
            file_size = emb_path.stat().st_size
            if shape[0] != processed_rows:
                encoder_errors.append(
                    f"Rows mismatch: embedding has {shape[0]}, processed parquet has {processed_rows}"
                )

        if not meta_path.exists():
            encoder_errors.append(f"Missing embedding meta: {meta_path}")
        else:
            meta = read_json(meta_path)
            if not meta.get("complete", False):
                encoder_errors.append("Embedding meta is not complete")
            if shape is not None:
                if int(meta.get("total_rows", -1)) != processed_rows:
                    encoder_errors.append(
                        f"Meta total_rows mismatch: {meta.get('total_rows')} != {processed_rows}"
                    )
                if int(meta.get("rows_written", -1)) != processed_rows:
                    encoder_errors.append(
                        f"Meta rows_written mismatch: {meta.get('rows_written')} != {processed_rows}"
                    )
                if int(meta.get("dim", -1)) != int(shape[1]):
                    encoder_errors.append(f"Meta dim mismatch: {meta.get('dim')} != {shape[1]}")

        encoder_results[key] = {
            "ok": not encoder_errors,
            "path": str(emb_path),
            "shape": shape,
            "file_size_bytes": int(file_size),
            "file_size_human": _human_bytes(file_size),
            "meta_path": str(meta_path),
            "meta": meta,
            "errors": encoder_errors,
        }
        errors.extend(f"{key}: {error}" for error in encoder_errors)

    summary = {
        "ok": not errors,
        "processed_path": str(processed_path),
        "processed_rows": int(processed_rows),
        "processed_columns": columns,
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "encoders": encoder_results,
        "errors": errors,
    }
    write_json(results_dir / "input_validation.json", summary)
    return summary


def _stratified_sample_indices(
    labels: np.ndarray,
    sample_size: int,
    random_state: int = 42,
) -> np.ndarray:
    labels = np.asarray(labels)
    n_rows = len(labels)
    if sample_size >= n_rows:
        return np.arange(n_rows, dtype="int64")

    rng = np.random.default_rng(random_state)
    unique_labels, counts = np.unique(labels, return_counts=True)
    expected = counts / counts.sum() * sample_size
    quotas = np.floor(expected).astype("int64")
    quotas = np.minimum(quotas, counts)

    remainder = sample_size - int(quotas.sum())
    if remainder > 0:
        order = np.argsort(expected - quotas)[::-1]
        for idx in order:
            if remainder <= 0:
                break
            capacity = counts[idx] - quotas[idx]
            if capacity <= 0:
                continue
            quotas[idx] += 1
            remainder -= 1

    sampled_parts: list[np.ndarray] = []
    selected_mask = np.zeros(n_rows, dtype=bool)
    for label, quota in zip(unique_labels, quotas):
        if quota <= 0:
            continue
        candidates = np.flatnonzero(labels == label)
        chosen = rng.choice(candidates, size=int(quota), replace=False)
        selected_mask[chosen] = True
        sampled_parts.append(chosen.astype("int64", copy=False))

    sampled = np.concatenate(sampled_parts) if sampled_parts else np.array([], dtype="int64")
    if len(sampled) < sample_size:
        remaining = np.flatnonzero(~selected_mask)
        fill = rng.choice(remaining, size=sample_size - len(sampled), replace=False)
        sampled = np.concatenate([sampled, fill.astype("int64", copy=False)])

    rng.shuffle(sampled)
    return sampled.astype("int64", copy=False)


def _label_codes(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes, uniques = pd.factorize(pd.Series(labels, dtype="object"), sort=True)
    return codes.astype("int32", copy=False), np.asarray(uniques, dtype=object)


HIGH_CONTRAST_COLORS = [
    "#E41A1C",  # red
    "#377EB8",  # blue
    "#4DAF4A",  # green
    "#984EA3",  # purple
    "#FF7F00",  # orange
    "#A65628",  # brown
    "#F781BF",  # pink
    "#999999",  # gray
    "#000000",  # black
    "#17BECF",  # cyan
]


def _high_contrast_palette(n_colors: int) -> list[str]:
    if n_colors <= len(HIGH_CONTRAST_COLORS):
        return HIGH_CONTRAST_COLORS[:n_colors]
    repeats = int(np.ceil(n_colors / len(HIGH_CONTRAST_COLORS)))
    return (HIGH_CONTRAST_COLORS * repeats)[:n_colors]


def _plot_umap_points(
    coords: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
    point_size: float = 2.0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codes, uniques = _label_codes(labels)
    colors = _high_contrast_palette(max(1, len(uniques)))
    cmap = ListedColormap(colors)

    plt.figure(figsize=(9, 7))
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=codes,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(uniques) - 0.5,
        s=point_size,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
    )
    plt.title(title)
    handles = [
        Line2D([0], [0], marker="o", color="w", label=str(label), markerfacecolor=colors[i], markersize=7)
        for i, label in enumerate(uniques)
    ]
    plt.legend(handles=handles, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_umap_grid(
    coords_by_encoder: dict[str, np.ndarray],
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
    point_size: float = 1.0,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    if not coords_by_encoder:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codes, uniques = _label_codes(labels)
    colors = _high_contrast_palette(max(1, len(uniques)))
    cmap = ListedColormap(colors)
    encoders = list(coords_by_encoder)
    ncols = min(3, len(encoders))
    nrows = int(np.ceil(len(encoders) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)

    for ax, encoder in zip(axes.ravel(), encoders):
        coords = coords_by_encoder[encoder]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=codes,
            cmap=cmap,
            vmin=-0.5,
            vmax=len(uniques) - 0.5,
            s=point_size,
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(encoder.upper())
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes.ravel()[len(encoders) :]:
        ax.axis("off")

    handles = [
        Line2D([0], [0], marker="o", color="w", label=str(label), markerfacecolor=colors[i], markersize=7)
        for i, label in enumerate(uniques)
    ]
    fig.suptitle(title, fontsize=14)
    fig.legend(handles=handles, fontsize=8, loc="center right")
    fig.tight_layout(rect=(0, 0, 0.88, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_full_umap_comparison(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_dir: str | Path = EMBEDDING_DIR,
    results_dir: str | Path = EXPERIMENTS_DIR / "server_full" / "umap",
    figures_dir: str | Path = EXPERIMENTS_DIR / "server_full" / "figures",
    clustering_dir: str | Path = EXPERIMENTS_DIR / "server_full" / "clustering",
    encoders: list[str] | None = None,
    label_column: str = "category",
    sample_size: int = 50_000,
    random_state: int = 42,
    reuse_existing_coords: bool = True,
) -> dict:
    import umap

    processed_path = Path(processed_path)
    embedding_dir = Path(embedding_dir)
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    clustering_dir = Path(clustering_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    encoders = encoders or ["tfidf", "w2v", "glove", "sbert", "bge"]

    meta_columns = [label_column, "rating"]
    meta = pd.read_parquet(processed_path, columns=meta_columns)
    labels = meta[label_column].astype("string").fillna("unknown").to_numpy()
    sample_indices = _stratified_sample_indices(labels, sample_size=sample_size, random_state=random_state)
    np.save(results_dir / "umap_sample_indices.npy", sample_indices)
    sample_meta = meta.iloc[sample_indices].copy()
    sample_meta.insert(0, "row_index", sample_indices)
    sample_meta.to_csv(results_dir / "umap_sample_metadata.csv", index=False)

    sampled_labels = labels[sample_indices]
    coords_by_encoder: dict[str, np.ndarray] = {}
    cluster_coords_by_encoder: dict[str, np.ndarray] = {}
    cluster_labels_by_encoder: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}

    for encoder in encoders:
        emb_path = embedding_dir / f"{encoder}.npy"
        coords_path = results_dir / f"{encoder}_umap_coords.npy"
        category_plot = figures_dir / f"{encoder}_umap_category_50k.png"
        cluster_plot = figures_dir / f"{encoder}_umap_clusters_50k.png"
        reused_coords = False
        if reuse_existing_coords and coords_path.exists():
            existing_coords = np.load(coords_path, mmap_mode="r")
            if existing_coords.shape == (len(sample_indices), 2):
                coords = np.asarray(existing_coords, dtype="float32")
                reused_coords = True
            else:
                del existing_coords
                coords_path.unlink()

        if not reused_coords:
            X = load_embedding(emb_path)
            if X.shape[0] != len(labels):
                raise ValueError(f"{encoder} rows {X.shape[0]} do not match labels {len(labels)}")
            vectors = np.asarray(X[sample_indices], dtype="float32", order="C")
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(vectors).astype("float32", copy=False)
            np.save(coords_path, coords)
            del X, vectors

        _plot_umap_points(coords, sampled_labels, category_plot, f"{encoder.upper()} UMAP by Category")
        coords_by_encoder[encoder] = coords

        cluster_path = clustering_dir / f"{encoder}_cluster_assignments.npy"
        cluster_plot_value = None
        if cluster_path.exists():
            cluster_assignments = np.load(cluster_path, mmap_mode="r")
            sampled_clusters = np.asarray(cluster_assignments[sample_indices], dtype="int32")
            _plot_umap_points(coords, sampled_clusters, cluster_plot, f"{encoder.upper()} UMAP by KMeans Cluster")
            cluster_coords_by_encoder[encoder] = coords
            cluster_labels_by_encoder[encoder] = sampled_clusters
            cluster_plot_value = str(cluster_plot)
            del cluster_assignments

        summary[encoder.upper()] = {
            "embedding_path": str(emb_path),
            "coords_path": str(coords_path),
            "category_plot": str(category_plot),
            "cluster_plot": cluster_plot_value,
            "sample_size": int(len(sample_indices)),
            "stratified_by": label_column,
            "random_state": random_state,
            "reused_coords": reused_coords,
        }
        del coords
        gc.collect()

    category_grid = figures_dir / "all_encoders_umap_category_grid.png"
    _plot_umap_grid(coords_by_encoder, sampled_labels, category_grid, "UMAP Comparison by Category")

    cluster_grid = figures_dir / "all_encoders_umap_cluster_grid.png"
    if len(cluster_labels_by_encoder) == len(coords_by_encoder):
        # Cluster ids are local to each encoder, so this grid compares cluster geometry per panel.
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.lines import Line2D

        encoders_with_clusters = list(cluster_coords_by_encoder)
        ncols = min(3, len(encoders_with_clusters))
        nrows = int(np.ceil(len(encoders_with_clusters) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows), squeeze=False)
        for ax, encoder in zip(axes.ravel(), encoders_with_clusters):
            cluster_labels = cluster_labels_by_encoder[encoder]
            codes, uniques = _label_codes(cluster_labels)
            colors = _high_contrast_palette(max(1, len(uniques)))
            cmap = ListedColormap(colors)
            coords = cluster_coords_by_encoder[encoder]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=codes,
                cmap=cmap,
                vmin=-0.5,
                vmax=len(uniques) - 0.5,
                s=1.0,
                alpha=0.65,
                linewidths=0,
                rasterized=True,
            )
            ax.set_title(encoder.upper())
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in axes.ravel()[len(encoders_with_clusters) :]:
            ax.axis("off")
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=f"cluster {i}",
                markerfacecolor=_high_contrast_palette(10)[i],
                markersize=7,
            )
            for i in range(min(10, len(np.unique(next(iter(cluster_labels_by_encoder.values()))))))
        ]
        fig.suptitle("UMAP Comparison by MiniBatchKMeans Cluster", fontsize=14)
        fig.legend(handles=handles, fontsize=8, loc="center right")
        fig.tight_layout(rect=(0, 0, 0.88, 0.95))
        fig.savefig(cluster_grid, dpi=180)
        plt.close(fig)

    all_summary = {
        "sample_indices_path": str(results_dir / "umap_sample_indices.npy"),
        "sample_metadata_path": str(results_dir / "umap_sample_metadata.csv"),
        "sample_size": int(len(sample_indices)),
        "stratified_by": label_column,
        "random_state": random_state,
        "category_grid": str(category_grid),
        "cluster_grid": str(cluster_grid) if cluster_grid.exists() else None,
        "encoders": summary,
    }
    write_json(results_dir / "all_umap_summary.json", all_summary)
    return all_summary


def run_full_linear_probe_for_embeddings(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_dir: str | Path = EMBEDDING_DIR,
    results_dir: str | Path = EXPERIMENTS_DIR / "server_full" / "linear_probe",
    encoders: list[str] | None = None,
    target_column: str = "rating",
    test_size: float = 0.2,
    batch_size: int = 16_384,
    epochs: int = 3,
    random_state: int = 42,
) -> dict:
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split

    processed_path = Path(processed_path)
    embedding_dir = Path(embedding_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    encoders = encoders or ["tfidf", "w2v", "glove", "sbert", "bge"]

    target_series = pd.read_parquet(processed_path, columns=[target_column])[target_column]
    valid_mask = target_series.notna().to_numpy()
    valid_indices = np.flatnonzero(valid_mask).astype("int64")
    y_all = target_series.loc[valid_mask].astype(int).to_numpy()
    train_indices, test_indices, y_train, y_test = train_test_split(
        valid_indices,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all,
    )
    classes = np.sort(np.unique(y_all))
    all_metrics: dict[str, dict] = {}

    for encoder in encoders:
        emb_path = embedding_dir / f"{encoder}.npy"
        X = load_embedding(emb_path)
        if X.shape[0] != len(target_series):
            raise ValueError(f"{encoder} rows {X.shape[0]} do not match target rows {len(target_series)}")

        clf = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            tol=None,
            random_state=random_state,
        )
        rng = np.random.default_rng(random_state)
        first_batch = True
        start_time = time.perf_counter()
        for epoch in range(epochs):
            order = rng.permutation(len(train_indices))
            epoch_indices = train_indices[order]
            epoch_targets = y_train[order]
            for start, end in tqdm(
                list(_batch_slices(len(epoch_indices), batch_size)),
                desc=f"Linear probe {encoder} epoch {epoch + 1}",
            ):
                batch_indices = epoch_indices[start:end]
                batch_y = epoch_targets[start:end]
                batch_X = np.asarray(X[batch_indices], dtype="float32", order="C")
                if first_batch:
                    clf.partial_fit(batch_X, batch_y, classes=classes)
                    first_batch = False
                else:
                    clf.partial_fit(batch_X, batch_y)

        predictions = np.empty(len(test_indices), dtype=classes.dtype)
        for start, end in tqdm(list(_batch_slices(len(test_indices), batch_size)), desc=f"Predict probe {encoder}"):
            batch_indices = test_indices[start:end]
            batch_X = np.asarray(X[batch_indices], dtype="float32", order="C")
            predictions[start:end] = clf.predict(batch_X)

        elapsed = time.perf_counter() - start_time
        report = classification_report(y_test, predictions, labels=classes, output_dict=True, zero_division=0)
        matrix = confusion_matrix(y_test, predictions, labels=classes)
        matrix_path = results_dir / f"{encoder}_confusion_matrix.csv"
        matrix_frame = pd.DataFrame(
            matrix,
            index=[f"true_{label}" for label in classes],
            columns=[f"pred_{label}" for label in classes],
        )
        matrix_frame.to_csv(matrix_path)

        metrics = {
            "embedding_path": str(emb_path),
            "target_column": target_column,
            "model": "SGDClassifier(loss=log_loss, penalty=l2)",
            "rows": int(len(target_series)),
            "valid_rows": int(len(valid_indices)),
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "test_size": test_size,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "classes": [int(label) for label in classes],
            "accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "per_class_f1": {
                str(label): float(report[str(label)]["f1-score"]) for label in classes if str(label) in report
            },
            "confusion_matrix_csv": str(matrix_path),
            "elapsed_seconds": float(elapsed),
        }
        write_json(results_dir / f"{encoder}_linear_probe_metrics.json", metrics)
        all_metrics[encoder.upper()] = metrics
        del X, predictions
        gc.collect()

    all_metrics_path = results_dir / "all_linear_probe_metrics.json"
    merged_metrics = read_json(all_metrics_path) if all_metrics_path.exists() else {}
    merged_metrics.update(all_metrics)
    write_json(all_metrics_path, merged_metrics)
    return merged_metrics


def _duration_to_seconds(value: str) -> int | None:
    parts = value.split(":")
    if not all(part.isdigit() for part in parts):
        return None
    numbers = [int(part) for part in parts]
    if len(numbers) == 3:
        hours, minutes, seconds = numbers
    elif len(numbers) == 2:
        hours, minutes, seconds = 0, numbers[0], numbers[1]
    else:
        return None
    return hours * 3600 + minutes * 60 + seconds


def _parse_encoder_runtime_from_logs(logs_dir: str | Path, encoder: str) -> dict | None:
    logs_dir = Path(logs_dir)
    if not logs_dir.exists():
        return None
    patterns = {
        "tfidf": ["Transform TF-IDF"],
        "w2v": ["Encode pretrained W2V", "Encode W2V", "Encode w2v"],
        "glove": ["Encode GloVe", "Encode glove"],
        "sbert": ["Encode sbert"],
        "bge": ["Encode bge"],
    }
    labels = patterns.get(encoder, [f"Encode {encoder}"])
    best: dict | None = None
    for log_path in sorted(logs_dir.glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore").replace("\r", "\n")
        for line in text.splitlines():
            if not any(label in line for label in labels) or "100%" not in line:
                continue
            match = re.search(r"\[([0-9:]+)<00:00,\s*([0-9.]+)(?:it|chunk)/s\]", line)
            if not match:
                continue
            elapsed_text = match.group(1)
            elapsed_seconds = _duration_to_seconds(elapsed_text)
            best = {
                "elapsed": elapsed_text,
                "elapsed_seconds": elapsed_seconds,
                "throughput_per_second": float(match.group(2)),
                "log_path": str(log_path),
            }
    return best


def run_full_efficiency_summary(
    processed_path: str | Path = PROCESSED_PATH,
    embedding_dir: str | Path = EMBEDDING_DIR,
    results_dir: str | Path = EXPERIMENTS_DIR / "server_full" / "efficiency",
    logs_dir: str | Path = PROJECT_ROOT / "logs",
    encoders: list[str] | None = None,
) -> dict:
    processed_path = Path(processed_path)
    embedding_dir = Path(embedding_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    encoders = encoders or ["tfidf", "w2v", "glove", "sbert", "bge"]
    processed_rows = parquet_row_count(processed_path)
    processed_summary_path = processed_path.with_name(processed_path.stem + "_summary.json")
    processed_summary = read_json(processed_summary_path) if processed_summary_path.exists() else {}
    embedding_run_summary_path = embedding_dir / "embedding_run_summary.json"
    embedding_run_summary = read_json(embedding_run_summary_path) if embedding_run_summary_path.exists() else {}

    encoder_metrics: dict[str, dict] = {}
    for encoder in encoders:
        emb_path = embedding_dir / f"{encoder}.npy"
        meta_path = _embedding_meta_path(emb_path)
        arr = np.load(emb_path, mmap_mode="r")
        shape = list(arr.shape)
        del arr
        file_size = emb_path.stat().st_size
        runtime = _parse_encoder_runtime_from_logs(logs_dir, encoder)
        run_summary = embedding_run_summary.get(encoder, {})
        encoder_metrics[encoder.upper()] = {
            "embedding_path": str(emb_path),
            "meta_path": str(meta_path),
            "meta": read_json(meta_path) if meta_path.exists() else {},
            "shape": shape,
            "rows": int(shape[0]),
            "dim": int(shape[1]),
            "file_size_bytes": int(file_size),
            "file_size_human": _human_bytes(file_size),
            "bytes_per_row": float(file_size / max(1, shape[0])),
            "device": run_summary.get("device"),
            "parsed_runtime": runtime,
        }

    summary_path = results_dir / "all_efficiency_metrics.json"
    existing_summary = read_json(summary_path) if summary_path.exists() else {}
    merged_encoder_metrics = dict(existing_summary.get("encoders", {}))
    merged_encoder_metrics.update(encoder_metrics)

    summary = {
        "processed_path": str(processed_path),
        "processed_rows": int(processed_rows),
        "processed_summary": processed_summary,
        "embedding_run_summary_path": str(embedding_run_summary_path),
        "logs_dir": str(logs_dir),
        "encoders": merged_encoder_metrics,
    }
    write_json(summary_path, summary)
    return summary


def _load_json_if_exists(path: str | Path) -> dict:
    path = Path(path)
    return read_json(path) if path.exists() else {}


def _metric_lookup(section: dict, encoder_key: str, *keys):
    value = section.get(encoder_key, {})
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [[str(value) if pd.notna(value) else "" for value in row] for row in frame.to_numpy()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def write_final_analysis_summary(
    results_root: str | Path = EXPERIMENTS_DIR / "server_full",
    encoders: list[str] | None = None,
) -> dict:
    results_root = Path(results_root)
    summary_dir = results_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    encoders = encoders or ["tfidf", "w2v", "glove", "sbert", "bge"]

    clustering = _load_json_if_exists(results_root / "clustering" / "all_clustering_metrics.json")
    retrieval = _load_json_if_exists(results_root / "retrieval" / "all_retrieval_metrics.json")
    anomaly = _load_json_if_exists(results_root / "anomaly" / "all_anomaly_metrics.json")
    linear_probe = _load_json_if_exists(results_root / "linear_probe" / "all_linear_probe_metrics.json")
    efficiency = _load_json_if_exists(results_root / "efficiency" / "all_efficiency_metrics.json")
    umap_summary = _load_json_if_exists(results_root / "umap" / "all_umap_summary.json")
    validation = _load_json_if_exists(results_root / "validation" / "input_validation.json")

    rows = []
    for encoder in encoders:
        key = encoder.upper()
        efficiency_metrics = efficiency.get("encoders", {}).get(key, {})
        umap_metrics = umap_summary.get("encoders", {}).get(key, {})
        rows.append(
            {
                "encoder": key,
                "rows": efficiency_metrics.get("rows"),
                "dim": efficiency_metrics.get("dim"),
                "file_size": efficiency_metrics.get("file_size_human"),
                "device": efficiency_metrics.get("device"),
                "embed_elapsed": (efficiency_metrics.get("parsed_runtime") or {}).get("elapsed"),
                "cluster_nmi": _metric_lookup(clustering, key, "nmi"),
                "cluster_ari": _metric_lookup(clustering, key, "ari"),
                "silhouette": _metric_lookup(clustering, key, "silhouette"),
                "retrieval_mrr": _metric_lookup(retrieval, key, "mrr"),
                "precision@5": _metric_lookup(retrieval, key, "precision@5"),
                "precision@10": _metric_lookup(retrieval, key, "precision@10"),
                "precision@50": _metric_lookup(retrieval, key, "precision@50"),
                "iso_anomalies": _metric_lookup(anomaly, key, "iso_anomaly_count"),
                "inconsistencies": _metric_lookup(anomaly, key, "inconsistency_count"),
                "linear_accuracy": _metric_lookup(linear_probe, key, "accuracy"),
                "linear_macro_f1": _metric_lookup(linear_probe, key, "macro_f1"),
                "linear_weighted_f1": _metric_lookup(linear_probe, key, "weighted_f1"),
                "umap_sample_size": umap_metrics.get("sample_size"),
                "umap_coords": umap_metrics.get("coords_path"),
            }
        )

    table = pd.DataFrame(rows)
    json_summary = {
        "validation": validation,
        "clustering": clustering,
        "retrieval": retrieval,
        "anomaly": anomaly,
        "linear_probe": linear_probe,
        "efficiency": efficiency,
        "umap": umap_summary,
        "table": rows,
    }
    write_json(summary_dir / "final_analysis_summary.json", json_summary)
    table.to_csv(summary_dir / "final_analysis_summary.csv", index=False)

    markdown = [
        "# Final Full Analysis Summary",
        "",
        f"Validation status: {'OK' if validation.get('ok') else 'MISSING_OR_FAILED'}",
        "",
        _markdown_table(table),
        "",
        "Notes:",
        "- Embeddings are reused from existing `.npy` artifacts; this summary does not rerun embedding generation.",
        "- Anomaly counts are controlled by contamination and should not be read as an encoder ranking by themselves.",
        "- Rating inconsistency uses full-data L2 SGD residuals for scalability.",
    ]
    (summary_dir / "final_analysis_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return json_summary
