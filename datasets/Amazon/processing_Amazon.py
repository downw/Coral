"""
Unified Amazon data processing pipeline.

This script merges the two original steps:
1. Raw Amazon review and metadata filtering from Core_filter.ipynb.
2. Leave one out sequence partitioning from processing_Amazon.py.

Typical use:
    python amazon_unified_processing.py \
        --review-file All_Amazon_Review_5.json.gz \
        --meta-file All_Amazon_Meta.json.gz \
        --intermediate-csv amazon_recent_20m_dataset.csv \
        --output-dir processed_data

If the intermediate CSV already exists:
    python amazon_unified_processing.py \
        --skip-raw-build \
        --intermediate-csv amazon_recent_20m_dataset.csv \
        --output-dir processed_data
"""

import argparse
import csv
import gc
import gzip
import hashlib
import json
import os
import pickle
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


@contextmanager
def open_jsonl_gzip(path: str, double_gzip: bool = False):
    """Open a JSONL gzip file, supporting both single and double gzip formats."""
    if double_gzip:
        with gzip.open(path, "rb") as f_outer:
            with gzip.open(f_outer, mode="rt", encoding="utf-8") as f_inner:
                yield f_inner
    else:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield f


def iter_jsonl_gzip(path: str, desc: str, double_gzip: bool = False) -> Iterable[dict]:
    """Yield JSON objects from a compressed JSONL file."""
    with open_jsonl_gzip(path, double_gzip=double_gzip) as f:
        for line in tqdm(f, desc=desc):
            try:
                yield json.loads(line.strip())
            except Exception:
                continue


def detect_double_gzip(path: str) -> bool:
    """Return True when the first decompressed bytes look like another gzip stream."""
    with gzip.open(path, "rb") as f:
        magic = f.read(2)
    return magic == b"\x1f\x8b"


def scan_cutoff_timestamp(review_file: str, target_size: int) -> int:
    """Find the timestamp cutoff needed to keep the most recent target_size reviews."""
    print("Phase 1: scanning timestamps to find cutoff time...")
    all_timestamps: List[int] = []

    for entry in iter_jsonl_gzip(review_file, desc="Reading timestamps"):
        ts = entry.get("unixReviewTime")
        if ts is not None:
            all_timestamps.append(int(ts))

    print(f"Total records found: {len(all_timestamps)}")

    if len(all_timestamps) > target_size:
        cutoff_index = len(all_timestamps) - target_size
        partitioned = np.partition(np.array(all_timestamps), cutoff_index)
        cutoff_time = int(partitioned[cutoff_index])
        del partitioned
        print(f"Cutoff timestamp determined: {cutoff_time}")
        print("Keeping data on or after this timestamp.")
    else:
        cutoff_time = 0
        print("Dataset size is smaller than target size. Keeping all records.")

    del all_timestamps
    gc.collect()
    return cutoff_time


def load_metadata_fields(
    meta_file: str,
    need_title: bool = False,
    double_gzip: Optional[bool] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load ASIN to category and, optionally, ASIN to title from Amazon metadata."""
    if double_gzip is None:
        double_gzip = detect_double_gzip(meta_file)

    category_map: Dict[str, str] = {}
    title_map: Dict[str, str] = {}
    mode = "double gzip" if double_gzip else "single gzip"
    print(f"Loading metadata from {meta_file} using {mode} mode...")

    for entry in iter_jsonl_gzip(meta_file, desc="Parsing metadata", double_gzip=double_gzip):
        asin = entry.get("asin")
        if not asin:
            continue

        categories = entry.get("category", [])
        category_map[asin] = categories[0] if categories else "Unknown"

        if need_title:
            title = entry.get("title")
            if title:
                title_map[asin] = title

    print(f"Loaded category information for {len(category_map)} items.")
    if need_title:
        print(f"Loaded titles for {len(title_map)} items.")
    return category_map, title_map


def build_recent_amazon_csv(
    review_file: str,
    meta_file: str,
    output_csv: str,
    target_size: int = 20_000_000,
    min_item_count: int = 5,
    min_seq_len: int = 20,
    batch_size: int = 1_000_000,
    include_title: bool = False,
    meta_double_gzip: Optional[bool] = None,
) -> None:
    """
    Build a filtered CSV from raw Amazon review and metadata files.

    The output columns are:
    user_id, item_id, timestamp, category, rating

    If include_title is True, a title column is appended.
    """
    cutoff_time = scan_cutoff_timestamp(review_file, target_size)

    print("\nPhase 2: loading recent reviews with chunking...")
    chunks: List[pd.DataFrame] = []
    batch_data: List[dict] = []

    for entry in iter_jsonl_gzip(review_file, desc="Loading recent reviews"):
        ts = entry.get("unixReviewTime")
        if ts is None or int(ts) < cutoff_time:
            continue

        review_text = entry.get("reviewText", "") or ""
        text_hash = hashlib.md5(review_text.encode("utf-8")).hexdigest()
        rating = entry.get("overall", 0.0)

        batch_data.append(
            {
                "user_id_raw": entry.get("reviewerID"),
                "item_id_raw": entry.get("asin"),
                "timestamp": int(ts),
                "text_hash": text_hash,
                "rating": rating,
            }
        )

        if len(batch_data) >= batch_size:
            chunk = pd.DataFrame(batch_data)
            chunk["timestamp"] = chunk["timestamp"].astype("int32")
            chunk["rating"] = chunk["rating"].astype("float32")
            chunks.append(chunk)
            batch_data = []
            gc.collect()

    if batch_data:
        chunk = pd.DataFrame(batch_data)
        chunk["timestamp"] = chunk["timestamp"].astype("int32")
        chunk["rating"] = chunk["rating"].astype("float32")
        chunks.append(chunk)
        del batch_data
        gc.collect()

    print(f"Loaded {len(chunks)} chunks. Concatenating...")
    if not chunks:
        raise ValueError("No review records were loaded after timestamp filtering.")

    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    print(f"Original raw shape: {df.shape}")

    print("Applying review deduplication...")
    df["review_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.date
    initial_len = len(df)
    df.drop_duplicates(subset=["user_id_raw", "review_date", "text_hash"], keep="first", inplace=True)
    print(f"Removed {initial_len - len(df)} duplicate reviews.")
    df.drop(columns=["review_date", "text_hash"], inplace=True)
    gc.collect()

    print("Filtering unpopular items...")
    item_counts = df["item_id_raw"].value_counts()
    valid_items = item_counts[item_counts >= min_item_count].index
    df = df[df["item_id_raw"].isin(valid_items)].copy()

    print("Filtering short user sequences...")
    user_counts = df["user_id_raw"].value_counts()
    valid_users = user_counts[user_counts >= min_seq_len].index
    df = df[df["user_id_raw"].isin(valid_users)].copy()
    print(f"Shape after filtering: {df.shape}")

    print("\nPhase 3: loading metadata and mapping categories...")
    category_map, title_map = load_metadata_fields(
        meta_file,
        need_title=include_title,
        double_gzip=meta_double_gzip,
    )
    df["category"] = df["item_id_raw"].map(category_map).fillna("Unknown")
    if include_title:
        df["title"] = df["item_id_raw"].map(title_map).fillna("Unknown")

    del category_map
    del title_map
    gc.collect()

    print("\nPhase 4: final reindexing and CSV saving...")
    df.sort_values(by=["user_id_raw", "timestamp", "item_id_raw"], inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    unique_users = df["user_id_raw"].unique()
    unique_items = df["item_id_raw"].unique()
    user_map = {u: i + 1 for i, u in enumerate(unique_users)}
    item_map = {item: i + 1 for i, item in enumerate(unique_items)}

    df["user_id"] = df["user_id_raw"].map(user_map)
    df["item_id"] = df["item_id_raw"].map(item_map)

    final_cols = ["user_id", "item_id", "timestamp", "category", "rating"]
    if include_title:
        final_cols.append("title")
    final_df = df[final_cols]

    print("\nFiltered CSV statistics:")
    print(f"Total users: {final_df['user_id'].nunique()}")
    print(f"Total items: {final_df['item_id'].nunique()}")
    print(f"Total interactions: {len(final_df)}")

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    print(f"Saving CSV to {output_csv}...")
    final_df.to_csv(output_csv, index=False)
    print("CSV generation done.")


def process_and_partition_amazon(
    csv_file: str,
    output_dir: str = "processed_data",
    k_core_threshold: int = 5,
) -> None:
    """Apply k core filtering to the CSV and create train, valid and test pickle files."""
    print(f"\nStart partitioning Amazon dataset from {csv_file}")
    print(f"Partition k core threshold: {k_core_threshold}")

    raw_interactions: List[Tuple[str, str, str, str, float]] = []
    user_count = defaultdict(int)
    item_count = defaultdict(int)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"Header found: {header}")

        required_cols = ["user_id", "item_id", "timestamp", "category", "rating"]
        col_idx = {name: header.index(name) for name in required_cols if name in header}
        missing = [name for name in required_cols if name not in col_idx]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        count = 0
        for row in reader:
            if len(row) < len(header):
                continue

            u_origin = row[col_idx["user_id"]]
            i_origin = row[col_idx["item_id"]]
            t_origin = row[col_idx["timestamp"]]
            c_origin = row[col_idx["category"]]
            try:
                r_origin = float(row[col_idx["rating"]])
            except ValueError:
                r_origin = 0.0

            user_count[u_origin] += 1
            item_count[i_origin] += 1
            raw_interactions.append((u_origin, i_origin, t_origin, c_origin, r_origin))

            count += 1
            if count % 1_000_000 == 0:
                print(f"Read {count} rows...")

    print(f"Raw CSV loading finished. Total rows: {len(raw_interactions)}")

    user2id: Dict[str, int] = {}
    item2id: Dict[str, int] = {}
    cat2id: Dict[str, int] = {}

    valid_users = {u for u, c in user_count.items() if c >= k_core_threshold}
    valid_items = {i for i, c in item_count.items() if c >= k_core_threshold}

    print(f"Before partition filtering: users={len(user_count)}, items={len(item_count)}")
    print(f"After partition filtering: users={len(valid_users)}, items={len(valid_items)}")

    user_interactions = defaultdict(list)
    skipped_count = 0

    for u_origin, i_origin, t_origin, c_origin, r_origin in raw_interactions:
        if u_origin not in valid_users or i_origin not in valid_items:
            skipped_count += 1
            continue

        if u_origin not in user2id:
            user2id[u_origin] = len(user2id) + 1
        if i_origin not in item2id:
            item2id[i_origin] = len(item2id) + 1
        if c_origin not in cat2id:
            cat2id[c_origin] = len(cat2id) + 1

        u_int = user2id[u_origin]
        i_int = item2id[i_origin]
        c_int = cat2id[c_origin]
        user_interactions[u_int].append((t_origin, i_int, c_int, r_origin))

    print(f"Skipped interactions during partition filtering: {skipped_count}")
    print(f"Final valid interactions: {len(raw_interactions) - skipped_count}")

    print("Sorting by timestamp and creating leave one out splits...")
    data_store = {"train": {}, "valid": {}, "test": {}}

    for u_int, interactions in user_interactions.items():
        interactions.sort(key=lambda x: x[0])
        seq_times, seq_items, seq_cats, seq_ratings = zip(*interactions)
        seq_times = list(seq_times)
        seq_items = list(seq_items)
        seq_cats = list(seq_cats)
        seq_ratings = list(seq_ratings)
        n = len(seq_items)

        if n < 3:
            data_store["train"][u_int] = (seq_items, seq_cats, seq_times, seq_ratings)
            data_store["valid"][u_int] = ([], [], [], [])
            data_store["test"][u_int] = ([], [], [], [])
        else:
            data_store["train"][u_int] = (
                seq_items[:-2],
                seq_cats[:-2],
                seq_times[:-2],
                seq_ratings[:-2],
            )
            data_store["valid"][u_int] = (
                seq_items[-2],
                seq_cats[-2],
                seq_times[-2],
                seq_ratings[-2],
            )
            data_store["test"][u_int] = (
                seq_items[-1],
                seq_cats[-1],
                seq_times[-1],
                seq_ratings[-1],
            )

    num_users = len(user2id)
    num_items = len(item2id)
    total_inter = len(raw_interactions) - skipped_count

    if num_users > 0 and num_items > 0:
        sparsity = 1 - (total_inter / (num_users * num_items))
        avg_len = total_inter / num_users
        print("\nDataset statistics after partitioning:")
        print(f"User count: {num_users}")
        print(f"Item count: {num_items}")
        print(f"Total interactions: {total_inter}")
        print(f"Average sequence length: {avg_len:.2f}")
        print(f"Sparsity: {sparsity:.6f} ({sparsity * 100:.4f}%)")

    dump_to_pickle(data_store, output_dir)
    save_maps(user2id, item2id, cat2id, output_dir)


def dump_to_pickle(data_store: dict, output_dir: str) -> None:
    """Save train, validation and test data in the original six list format."""
    print("Generating pickle files in format [Item, User, Cat, Target, Time, Rating]...")
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "train": [[], [], [], [], [], []],
        "valid": [[], [], [], [], [], []],
        "test": [[], [], [], [], [], []],
    }

    users = sorted(data_store["train"].keys())

    for u in users:
        train_i, train_c, train_t, train_r = data_store["train"][u]
        valid_i, valid_c, valid_t, valid_r = data_store["valid"][u]
        test_i, test_c, test_t, test_r = data_store["test"][u]

        has_valid = valid_i != [] and isinstance(valid_i, int)
        has_test = test_i != [] and isinstance(test_i, int)

        datasets["train"][0].append(train_i)
        datasets["train"][1].append([u])
        datasets["train"][2].append(train_c)
        datasets["train"][3].append(valid_i if has_valid else 0)
        datasets["train"][4].append(train_t)
        datasets["train"][5].append(train_r)

        if has_valid:
            datasets["valid"][0].append(train_i)
            datasets["valid"][1].append([u])
            datasets["valid"][2].append(train_c)
            datasets["valid"][3].append(valid_i)
            datasets["valid"][4].append(train_t)
            datasets["valid"][5].append(train_r)

        if has_test:
            if has_valid:
                test_seq_i = train_i + [valid_i]
                test_seq_c = train_c + [valid_c]
                test_seq_t = train_t + [valid_t]
                test_seq_r = train_r + [valid_r]
            else:
                test_seq_i = train_i
                test_seq_c = train_c
                test_seq_t = train_t
                test_seq_r = train_r

            datasets["test"][0].append(test_seq_i)
            datasets["test"][1].append([u])
            datasets["test"][2].append(test_seq_c)
            datasets["test"][3].append(test_i)
            datasets["test"][4].append(test_seq_t)
            datasets["test"][5].append(test_seq_r)

    for mode, data_list in datasets.items():
        path = os.path.join(output_dir, f"{mode}.txt")
        with open(path, "wb") as f:
            pickle.dump(data_list, f)
        print(f"Saved {path}. Number of samples: {len(data_list[0])}")


def save_maps(user2id: dict, item2id: dict, cat2id: dict, output_dir: str) -> None:
    """Save ID mappings and basic statistics."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "id_map.pickle"), "wb") as f:
        pickle.dump({"user2id": user2id, "item2id": item2id, "cat2id": cat2id}, f)

    with open(os.path.join(output_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"num_users={len(user2id)}\n")
        f.write(f"num_items={len(item2id)}\n")
        f.write(f"num_cats={len(cat2id)}\n")

    print("ID mappings and statistics files have been saved.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Amazon preprocessing and partitioning pipeline.")
    parser.add_argument("--review-file", default="All_Amazon_Review_5.json.gz")
    parser.add_argument("--meta-file", default="All_Amazon_Meta.json.gz")
    parser.add_argument("--intermediate-csv", default="amazon_recent_20m_dataset.csv")
    parser.add_argument("--output-dir", default="processed_data")
    parser.add_argument("--target-size", type=int, default=20_000_000)
    parser.add_argument("--min-item-count", type=int, default=5)
    parser.add_argument("--min-seq-len", type=int, default=20)
    parser.add_argument("--k-core-threshold", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--include-title", action="store_true")
    parser.add_argument("--skip-raw-build", action="store_true")
    parser.add_argument("--only-build-csv", action="store_true")
    parser.add_argument(
        "--meta-compression",
        choices=["auto", "single", "double"],
        default="auto",
        help="Metadata gzip format. The original notebook used double gzip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    meta_double_gzip = None
    if args.meta_compression == "single":
        meta_double_gzip = False
    elif args.meta_compression == "double":
        meta_double_gzip = True

    if not args.skip_raw_build:
        build_recent_amazon_csv(
            review_file=args.review_file,
            meta_file=args.meta_file,
            output_csv=args.intermediate_csv,
            target_size=args.target_size,
            min_item_count=args.min_item_count,
            min_seq_len=args.min_seq_len,
            batch_size=args.batch_size,
            include_title=args.include_title,
            meta_double_gzip=meta_double_gzip,
        )
    else:
        if not os.path.exists(args.intermediate_csv):
            raise FileNotFoundError(f"Intermediate CSV not found: {args.intermediate_csv}")
        print(f"Skipping raw build. Using existing CSV: {args.intermediate_csv}")

    if not args.only_build_csv:
        process_and_partition_amazon(
            csv_file=args.intermediate_csv,
            output_dir=args.output_dir,
            k_core_threshold=args.k_core_threshold,
        )


if __name__ == "__main__":
    main()
