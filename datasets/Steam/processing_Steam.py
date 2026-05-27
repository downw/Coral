#!/usr/bin/env python3
"""
Unified Steam preprocessing script.

This script merges the raw Steam filtering pipeline from Core_filter(2).ipynb
with the train/valid/test partitioning pipeline from processing_Steam.py.

Pipeline:
  1. Read raw Steam reviews and metadata.
  2. Apply user and item frequency filtering.
  3. Build a sequential CSV with columns:
       user_id, item_id, timestamp, category, hours
  4. Optionally map complex Steam genre/category strings to coarser categories.
  5. Partition the mapped CSV into train.txt, valid.txt, test.txt, id_map.pickle.

Output pickle format is kept consistent with the original processing_Steam.py:
  [Inputs, User, Cats, Targets, Times, Hours]
"""

import argparse
import ast
import csv
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm is convenient but should not be mandatory
    tqdm = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def iter_with_progress(iterable: Iterable, total: Optional[int] = None, unit: str = "it"):
    """Wrap an iterable with tqdm when tqdm is available."""
    if tqdm is not None:
        return tqdm(iterable, total=total, unit=unit, mininterval=0.5)
    return iterable


def parse_literal_line(line: str) -> Optional[dict]:
    """Parse Steam JSON-like lines stored as Python literal dictionaries."""
    try:
        obj = ast.literal_eval(line.strip())
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_steam_date(date_str: Optional[str]) -> Optional[pd.Timestamp]:
    """Parse date strings used in the Steam review dump."""
    if not date_str or pd.isna(date_str):
        return None

    candidates = []
    raw = str(date_str).strip()

    # Common forms observed in Steam review dumps include:
    #   Posted November 5, 2015.
    #   Posted: November 5, 2015
    #   November 5, 2015
    #   Posted November 5.  The notebook defaulted missing years to 2016.
    clean = raw.replace("Posted:", "").replace("Posted", "").replace(".", "").strip()
    if "Updated:" in clean:
        clean = clean.split("Updated:")[0].strip()

    candidates.append(clean)
    if "," not in clean:
        candidates.append(f"{clean}, 2016")

    for candidate in candidates:
        try:
            dt = pd.to_datetime(candidate, errors="coerce")
            if not pd.isna(dt):
                return dt
        except Exception:
            continue
    return None


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Raw Steam filtering and CSV construction
# ---------------------------------------------------------------------------


def load_game_metadata(meta_file: str, est_meta_lines: Optional[int] = None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load item to category and item to title mappings from steam_games.json."""
    print(f"Loading metadata from {meta_file}...")
    category_map: Dict[str, str] = {}
    title_map: Dict[str, str] = {}

    if not os.path.exists(meta_file):
        print(f"Warning: {meta_file} not found. Categories and titles will use fallback values.")
        return category_map, title_map

    with open(meta_file, "r", encoding="utf-8") as f:
        for line in iter_with_progress(f, total=est_meta_lines, unit="game"):
            entry = parse_literal_line(line)
            if not entry:
                continue

            raw_id = entry.get("id")
            if raw_id is None:
                continue
            item_id = str(raw_id)

            genres = entry.get("genres")
            tags = entry.get("tags")
            category_str = "Unknown"
            if isinstance(genres, list) and len(genres) > 0:
                category_str = "|".join(str(g) for g in genres)
            elif genres:
                category_str = str(genres)
            elif isinstance(tags, list) and len(tags) > 0:
                category_str = "|".join(str(t) for t in tags)
            elif tags:
                category_str = str(tags)
            category_map[item_id] = category_str

            title = entry.get("app_name") or entry.get("title")
            if title:
                title_map[item_id] = str(title)

    print(f"Loaded categories for {len(category_map)} games.")
    print(f"Loaded titles for {len(title_map)} games.")
    return category_map, title_map


def build_steam_sequential_csv(
    review_file: str,
    meta_file: str,
    output_csv: str = "steam_sequential_dataset.csv",
    min_user_interactions: int = 20,
    min_item_interactions: int = 5,
    est_review_lines: Optional[int] = 7_793_069,
    est_meta_lines: Optional[int] = 32_135,
    include_title: bool = False,
    title_csv: str = "steam_sequential_dataset_with_title.csv",
) -> None:
    """Build the filtered Steam sequential CSV from raw review and metadata files."""
    print("=" * 60)
    print("Building filtered Steam sequential CSV")
    print("=" * 60)

    category_map, title_map = load_game_metadata(meta_file, est_meta_lines=est_meta_lines)

    user_counter: Counter = Counter()
    item_counter: Counter = Counter()

    print("\nPhase 1: scanning review file to count user and item activity...")
    with open(review_file, "r", encoding="utf-8") as f:
        for line in iter_with_progress(f, total=est_review_lines, unit="line"):
            entry = parse_literal_line(line)
            if not entry:
                continue
            username = entry.get("username")
            raw_item_id = entry.get("product_id")
            if username:
                user_counter[str(username)] += 1
            if raw_item_id:
                item_counter[str(raw_item_id)] += 1

    valid_users = {u for u, c in user_counter.items() if c >= min_user_interactions}
    valid_items = {i for i, c in item_counter.items() if c >= min_item_interactions}

    print("\nFiltering thresholds:")
    print(f"  min_user_interactions = {min_user_interactions}")
    print(f"  min_item_interactions = {min_item_interactions}")
    print(f"Raw users: {len(user_counter):,} | Valid users: {len(valid_users):,}")
    print(f"Raw items: {len(item_counter):,} | Valid items: {len(valid_items):,}")

    rows: List[dict] = []

    print("\nPhase 2: extracting filtered interactions and matching categories...")
    with open(review_file, "r", encoding="utf-8") as f:
        for line in iter_with_progress(f, total=est_review_lines, unit="line"):
            entry = parse_literal_line(line)
            if not entry:
                continue

            username = entry.get("username")
            if not username or str(username) not in valid_users:
                continue

            raw_item_id = entry.get("product_id")
            if not raw_item_id:
                continue
            item_id_str = str(raw_item_id)
            if item_id_str not in valid_items:
                continue

            dt_obj = parse_steam_date(entry.get("date") or entry.get("date_posted"))
            if dt_obj is None or pd.isna(dt_obj):
                continue

            record = {
                "username_raw": str(username),
                "item_id_raw": item_id_str,
                "timestamp": dt_obj,
                "category": category_map.get(item_id_str, "Unknown"),
                "hours": safe_float(entry.get("hours", 0.0)),
            }
            if include_title:
                record["title"] = title_map.get(item_id_str, "Unknown")
            rows.append(record)

    print("\nPhase 3: creating DataFrame and reindexing user/item IDs...")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid Steam interactions were retained. Please check input files and thresholds.")

    df = df.sort_values(by=["username_raw", "timestamp"])

    user_map = {name: idx + 1 for idx, name in enumerate(df["username_raw"].unique())}
    item_map = {item: idx + 1 for idx, item in enumerate(df["item_id_raw"].unique())}
    df["user_id"] = df["username_raw"].map(user_map)
    df["item_id"] = df["item_id_raw"].map(item_map)

    base_cols = ["user_id", "item_id", "timestamp", "category", "hours"]
    final_df = df[base_cols].sort_values(by=["user_id", "timestamp"])

    print(f"\nPhase 4: saving sequential CSV to {output_csv}...")
    final_df.to_csv(output_csv, index=False)

    if include_title:
        title_cols = base_cols + ["title"]
        title_df = df[title_cols].sort_values(by=["user_id", "timestamp"])
        title_df.to_csv(title_csv, index=False)
        print(f"Saved title CSV to {title_csv}")

    print("=" * 60)
    print("Sequential CSV completed")
    print(f"Output path: {os.path.abspath(output_csv)}")
    print(f"Records: {len(final_df):,}")
    print(f"Users: {final_df['user_id'].nunique():,}")
    print(f"Items: {final_df['item_id'].nunique():,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Category mapping from complex Steam genre strings to coarser labels
# ---------------------------------------------------------------------------


def map_category(category_combination) -> str:
    """Map raw Steam genre/tag combinations into coarser category labels."""
    if pd.isna(category_combination) or category_combination == "Unknown":
        return "Other"

    cat_lower = str(category_combination).lower()

    combo_categories = {
        ("action", "adventure"): "Action-Adventure",
        ("action", "rpg"): "Action-RPG",
        ("strategy", "rpg"): "Strategy-RPG",
        ("racing", "simulation"): "Racing-Sim",
        ("sports", "simulation"): "Sports-Sim",
        ("tower defense", "strategy"): "Tower Defense",
        ("visual novel", "rpg"): "Visual Novel",
    }
    for combo_keys, combo_name in combo_categories.items():
        if all(key in cat_lower for key in combo_keys):
            return combo_name

    primary_categories = {
        "utilities": "Utilities",
        "animation & modeling": "Animation & Modeling",
        "design & illustration": "Design & Illustration",
        "video production": "Video Production",
        "audio production": "Audio Production",
        "software training": "Software Training",
        "web publishing": "Web Publishing",
        "education": "Education",
        "game development": "Game Development",
        "photo editing": "Photo Editing",
        "racing": "Racing",
        "sports": "Sports",
        "flight": "Simulation",
        "puzzle": "Puzzle",
        "music": "Music",
        "fighting": "Fighting",
        "horror": "Horror",
        "rts": "RTS",
        "moba": "MOBA",
        "tower defense": "Tower Defense",
        "card game": "Card Game",
        "visual novel": "Visual Novel",
        "strategy": "Strategy",
        "rpg": "RPG",
        "simulation": "Simulation",
        "tactical": "Strategy",
        "shooter": "Shooter",
        "fps": "Shooter",
        "platformer": "Platformer",
        "rogue": "Roguelike",
        "metroidvania": "Action-Adventure",
        "hack and slash": "Action",
        "massively multiplayer": "MMO",
        "action": "Action",
        "adventure": "Adventure",
        "casual": "Casual",
        "arcade": "Arcade",
        "anime": "Anime",
        "movie": "Movie",
        "episodic": "Episodic",
        "indie": "Indie",
    }
    for key, main_cat in primary_categories.items():
        if key in cat_lower:
            return main_cat

    parts = str(category_combination).split("|")
    if parts and parts[0].strip():
        fallback = parts[0].strip()
        if fallback in {"Early Access", "Free to Play"}:
            return "Other"
        return fallback

    return "Other"


def map_steam_categories(input_csv: str, output_csv: str = "steam_sequential_mapped.csv") -> None:
    """Apply category mapping and save the mapped CSV."""
    print("=" * 60)
    print("Mapping Steam categories")
    print("=" * 60)
    print(f"Loading {input_csv}...")

    df = pd.read_csv(input_csv)
    if "category" not in df.columns:
        raise ValueError("Input CSV must contain a 'category' column.")

    raw_unique = df["category"].nunique(dropna=False)
    df["category"] = df["category"].apply(map_category)
    mapped_unique = df["category"].nunique(dropna=False)

    df.to_csv(output_csv, index=False)
    print(f"Mapped categories: {raw_unique} raw types -> {mapped_unique} mapped types")
    print(f"Saved mapped CSV to {os.path.abspath(output_csv)}")


# ---------------------------------------------------------------------------
# Train/valid/test partitioning, kept consistent with processing_Steam.py
# ---------------------------------------------------------------------------


def process_and_partition_steam(csv_file: str, output_dir: str = "processed_data") -> None:
    print(f"--- Start processing file: {csv_file} ---")

    user2id: Dict[str, int] = {}
    item2id: Dict[str, int] = {}
    cat2id: Dict[str, int] = {}

    # UserID -> List[(Timestamp, ItemID, CatID, Hours)]
    user_interactions = defaultdict(list)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"Detected header: {header}")

        header_map = {name: idx for idx, name in enumerate(header)}
        required = ["user_id", "item_id", "timestamp", "category", "hours"]
        missing = [col for col in required if col not in header_map]
        if missing:
            raise ValueError(f"Missing required columns in {csv_file}: {missing}")

        count = 0
        for row in reader:
            if len(row) < len(header):
                continue

            u_origin = row[header_map["user_id"]]
            i_origin = row[header_map["item_id"]]
            t_origin = row[header_map["timestamp"]]
            c_origin = row[header_map["category"]]
            h_origin = safe_float(row[header_map["hours"]])

            if u_origin not in user2id:
                user2id[u_origin] = len(user2id) + 1
            if i_origin not in item2id:
                item2id[i_origin] = len(item2id) + 1
            if c_origin not in cat2id:
                cat2id[c_origin] = len(cat2id) + 1

            u_int = user2id[u_origin]
            i_int = item2id[i_origin]
            c_int = cat2id[c_origin]

            user_interactions[u_int].append((t_origin, i_int, c_int, h_origin))

            count += 1
            if count % 1_000_000 == 0:
                print(f"Processed {count:,} rows...")

    print(f"Finished reading. Total interactions: {count:,}")
    print(f"Users: {len(user2id):,}, Items: {len(item2id):,}, Categories: {len(cat2id):,}")

    print("Sorting by timestamp and splitting datasets...")
    data_store = {"train": {}, "valid": {}, "test": {}}

    for u_int, interactions in user_interactions.items():
        interactions.sort(key=lambda x: x[0])
        seq_times, seq_items, seq_cats, seq_hours = zip(*interactions)

        seq_times = list(seq_times)
        seq_items = list(seq_items)
        seq_cats = list(seq_cats)
        seq_hours = list(seq_hours)

        n = len(seq_items)
        if n < 3:
            data_store["train"][u_int] = (seq_items, seq_cats, seq_times, seq_hours)
            data_store["valid"][u_int] = ([], [], [], [])
            data_store["test"][u_int] = ([], [], [], [])
        else:
            data_store["train"][u_int] = (seq_items[:-2], seq_cats[:-2], seq_times[:-2], seq_hours[:-2])
            data_store["valid"][u_int] = (seq_items[-2], seq_cats[-2], seq_times[-2], seq_hours[-2])
            data_store["test"][u_int] = (seq_items[-1], seq_cats[-1], seq_times[-1], seq_hours[-1])

    dump_to_pickle(data_store, output_dir)
    save_maps(user2id, item2id, cat2id, output_dir)
    save_stats(user2id, item2id, cat2id, count, output_dir)


def dump_to_pickle(data_store: dict, output_dir: str) -> None:
    print("Generating Pickle files (structure: [Inputs, User, Cats, Targets, Times, Hours])...")
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "train": [[], [], [], [], [], []],
        "valid": [[], [], [], [], [], []],
        "test": [[], [], [], [], [], []],
    }

    users = sorted(data_store["train"].keys())

    for u in users:
        train_i, train_c, train_t, train_h = data_store["train"][u]
        valid_i, valid_c, valid_t, valid_h = data_store["valid"][u]
        test_i, test_c, test_t, test_h = data_store["test"][u]

        has_valid = valid_i != [] and isinstance(valid_i, int)
        has_test = test_i != [] and isinstance(test_i, int)

        datasets["train"][0].append(train_i)
        datasets["train"][1].append([u])
        datasets["train"][2].append(train_c)
        datasets["train"][3].append(valid_i if has_valid else 0)
        datasets["train"][4].append(train_t)
        datasets["train"][5].append(train_h)

        if has_valid:
            datasets["valid"][0].append(train_i)
            datasets["valid"][1].append([u])
            datasets["valid"][2].append(train_c)
            datasets["valid"][3].append(valid_i)
            datasets["valid"][4].append(train_t)
            datasets["valid"][5].append(train_h)

        if has_test:
            if has_valid:
                test_seq_i = train_i + [valid_i]
                test_seq_c = train_c + [valid_c]
                test_seq_t = train_t + [valid_t]
                test_seq_h = train_h + [valid_h]
            else:
                test_seq_i = train_i
                test_seq_c = train_c
                test_seq_t = train_t
                test_seq_h = train_h

            datasets["test"][0].append(test_seq_i)
            datasets["test"][1].append([u])
            datasets["test"][2].append(test_seq_c)
            datasets["test"][3].append(test_i)
            datasets["test"][4].append(test_seq_t)
            datasets["test"][5].append(test_seq_h)

    for mode, data_list in datasets.items():
        path = os.path.join(output_dir, f"{mode}.txt")
        with open(path, "wb") as f:
            pickle.dump(data_list, f)
        print(f"Saved {mode}.txt | Number of samples: {len(data_list[0]):,}")


def save_maps(user2id: dict, item2id: dict, cat2id: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "id_map.pickle"), "wb") as f:
        pickle.dump({"user2id": user2id, "item2id": item2id, "cat2id": cat2id}, f)
    print("ID mapping tables have been saved.")


def save_stats(user2id: dict, item2id: dict, cat2id: dict, num_interactions: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"num_users={len(user2id)}\n")
        f.write(f"num_items={len(item2id)}\n")
        f.write(f"num_cats={len(cat2id)}\n")
        f.write(f"num_interactions={num_interactions}\n")
    print("Statistics file has been saved.")


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Steam preprocessing pipeline")

    parser.add_argument("--review-file", default="steam.json", help="Raw Steam review file")
    parser.add_argument("--meta-file", default="steam_games.json", help="Raw Steam metadata file")
    parser.add_argument("--sequential-csv", default="steam_sequential_dataset.csv", help="Intermediate sequential CSV")
    parser.add_argument("--mapped-csv", default="steam_sequential_mapped.csv", help="Category-mapped CSV used for partitioning")
    parser.add_argument("--output-dir", default="processed_data", help="Directory for train/valid/test pickle files")

    parser.add_argument("--min-user-interactions", type=int, default=20, help="Minimum raw interactions per user")
    parser.add_argument("--min-item-interactions", type=int, default=5, help="Minimum raw interactions per item")
    parser.add_argument("--est-review-lines", type=int, default=7_793_069, help="Estimated review lines for progress bar")
    parser.add_argument("--est-meta-lines", type=int, default=32_135, help="Estimated metadata lines for progress bar")

    parser.add_argument("--skip-raw-build", action="store_true", help="Skip raw review/meta processing and use existing sequential CSV")
    parser.add_argument("--skip-category-mapping", action="store_true", help="Skip category mapping and partition the sequential CSV directly")
    parser.add_argument("--include-title", action="store_true", help="Also save steam_sequential_dataset_with_title.csv")
    parser.add_argument("--title-csv", default="steam_sequential_dataset_with_title.csv", help="Optional CSV with game titles")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_raw_build:
        if not os.path.exists(args.review_file):
            raise FileNotFoundError(f"Review file not found: {args.review_file}")
        if not os.path.exists(args.meta_file):
            raise FileNotFoundError(f"Metadata file not found: {args.meta_file}")

        build_steam_sequential_csv(
            review_file=args.review_file,
            meta_file=args.meta_file,
            output_csv=args.sequential_csv,
            min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions,
            est_review_lines=args.est_review_lines,
            est_meta_lines=args.est_meta_lines,
            include_title=args.include_title,
            title_csv=args.title_csv,
        )
    else:
        if not os.path.exists(args.sequential_csv):
            raise FileNotFoundError(f"Sequential CSV not found: {args.sequential_csv}")
        print(f"Skipping raw build. Using existing sequential CSV: {args.sequential_csv}")

    partition_input = args.sequential_csv
    if not args.skip_category_mapping:
        map_steam_categories(args.sequential_csv, args.mapped_csv)
        partition_input = args.mapped_csv
    else:
        print("Skipping category mapping. Partitioning sequential CSV directly.")

    process_and_partition_steam(partition_input, args.output_dir)


if __name__ == "__main__":
    main()
