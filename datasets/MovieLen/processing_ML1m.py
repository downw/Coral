#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified MovieLens 1M preprocessing script.

This file merges two steps into one executable Python script:

1. Build a sequential CSV from MovieLens raw files:
   - movies.dat
   - ratings.dat

2. Partition the generated CSV into the pickle format used by the model:
   - train.txt
   - valid.txt
   - test.txt
   - id_map.pickle
   - stats.txt

Default output format for train, valid, and test:
    [Item sequence, User ID, Category sequence, Target item, Time sequence, Rating sequence]

Typical usage:
    python movielens_unified_processing.py \
        --movies-file movies.dat \
        --ratings-file ratings.dat \
        --intermediate-csv movielens_sequential_dataset.csv \
        --output-dir processed_data

If the intermediate CSV already exists:
    python movielens_unified_processing.py \
        --skip-raw-build \
        --intermediate-csv movielens_sequential_dataset.csv \
        --output-dir processed_data
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import pandas as pd


def map_movielens_category(genre_str):
    """
    Map MovieLens combined genres to a single main category.

    The priority order follows the notebook logic: more specific genres are
    preferred over broader genres.
    """
    if pd.isna(genre_str) or genre_str == "(no genres listed)":
        return "Unknown"

    genres = str(genre_str).split("|")

    priority_map = [
        "Film-Noir",
        "Documentary",
        "War",
        "Western",
        "Musical",
        "Animation",
        "Children's",
        "Fantasy",
        "Sci-Fi",
        "Horror",
        "Crime",
        "Thriller",
        "Mystery",
        "Action",
        "Adventure",
        "Romance",
        "Comedy",
        "Drama",
    ]

    for genre in priority_map:
        if genre in genres:
            return genre

    return genres[0] if genres else "Unknown"


def build_movielens_sequential_csv(
    movies_file="movies.dat",
    ratings_file="ratings.dat",
    output_file="movielens_sequential_dataset.csv",
    min_item_count=5,
    min_seq_len=20,
    include_title=False,
    title_output_file="movielens_sequential_dataset_with_title.csv",
):
    """
    Build a cleaned sequential MovieLens CSV from the original MovieLens 1M files.

    Input files should follow MovieLens 1M format:
        movies.dat:  MovieID::Title::Genres
        ratings.dat: UserID::MovieID::Rating::Timestamp

    The produced main CSV columns are:
        user_id, item_id, rating, timestamp, category
    """
    print("--- Building MovieLens sequential CSV ---")
    print(f"Movies file: {movies_file}")
    print(f"Ratings file: {ratings_file}")
    print(f"Output CSV: {output_file}")
    print(f"Min item count: {min_item_count}")
    print(f"Min user sequence length: {min_seq_len}")

    if not os.path.exists(movies_file):
        raise FileNotFoundError(f"Movies file not found: {movies_file}")
    if not os.path.exists(ratings_file):
        raise FileNotFoundError(f"Ratings file not found: {ratings_file}")

    print("Loading movies data...")
    movies_df = pd.read_csv(
        movies_file,
        sep="::",
        header=None,
        names=["MovieID", "Title", "Genres"],
        engine="python",
        encoding="latin-1",
    )

    movies_df["category"] = movies_df["Genres"].apply(map_movielens_category)
    item_category_map = dict(zip(movies_df["MovieID"], movies_df["category"]))
    item_title_map = dict(zip(movies_df["MovieID"], movies_df["Title"]))

    print(f"Loaded movies: {len(movies_df)}")

    print("Loading ratings data...")
    ratings_df = pd.read_csv(
        ratings_file,
        sep="::",
        header=None,
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        engine="python",
    )

    ratings_df = ratings_df[["UserID", "MovieID", "Rating", "Timestamp"]]

    print("Filtering unpopular items...")
    original_len = len(ratings_df)
    item_counts = ratings_df["MovieID"].value_counts()
    valid_items = item_counts[item_counts >= min_item_count].index
    ratings_df = ratings_df[ratings_df["MovieID"].isin(valid_items)].copy()

    print(f"Original interactions: {original_len}")
    print(f"After item filtering: {len(ratings_df)}")

    print("Filtering short user sequences...")
    user_counts = ratings_df["UserID"].value_counts()
    valid_users = user_counts[user_counts >= min_seq_len].index
    ratings_df = ratings_df[ratings_df["UserID"].isin(valid_users)].copy()

    print(f"After user sequence filtering: {len(ratings_df)}")

    print("Remapping user and item IDs...")
    ratings_df.sort_values(by=["UserID", "Timestamp"], inplace=True)

    unique_users = ratings_df["UserID"].unique()
    unique_items = ratings_df["MovieID"].unique()

    user_map = {uid: idx + 1 for idx, uid in enumerate(unique_users)}
    item_map = {iid: idx + 1 for idx, iid in enumerate(unique_items)}

    ratings_df["user_id"] = ratings_df["UserID"].map(user_map)
    ratings_df["item_id"] = ratings_df["MovieID"].map(item_map)
    ratings_df["rating"] = ratings_df["Rating"]
    ratings_df["timestamp"] = pd.to_datetime(ratings_df["Timestamp"], unit="s")
    ratings_df["category"] = ratings_df["MovieID"].map(item_category_map)

    final_df = ratings_df[["user_id", "item_id", "rating", "timestamp", "category"]].copy()

    print_statistics(final_df)

    print(f"Saving sequential CSV to {output_file}...")
    final_df.to_csv(output_file, index=False)

    if include_title:
        ratings_df["title"] = ratings_df["MovieID"].map(item_title_map)
        final_df_with_title = ratings_df[
            ["user_id", "item_id", "rating", "timestamp", "category", "title"]
        ].copy()
        print(f"Saving title enriched CSV to {title_output_file}...")
        final_df_with_title.to_csv(title_output_file, index=False)

    print("Sequential CSV generation completed.")
    return output_file


def print_statistics(final_df):
    """Print basic statistics for the sequential CSV."""
    print("\n" + "=" * 40)
    print("Dataset Statistics")
    print("=" * 40)

    n_users = final_df["user_id"].nunique()
    n_items = final_df["item_id"].nunique()
    n_interactions = len(final_df)

    seq_lens = final_df.groupby("user_id").size()
    avg_seq_len = seq_lens.mean() if len(seq_lens) > 0 else 0
    avg_unique_cats = (
        final_df.groupby("user_id")["category"].nunique().mean()
        if len(seq_lens) > 0
        else 0
    )

    sparsity = 1 - (n_interactions / (n_users * n_items)) if n_users and n_items else 0

    print(f"Total Users: {n_users}")
    print(f"Total Items: {n_items}")
    print(f"Total Interactions: {n_interactions}")
    print(f"Sparsity: {sparsity:.5f}")

    if len(seq_lens) > 0:
        print("-" * 30)
        print(f"Min Sequence Length: {seq_lens.min()}")
        print(f"Max Sequence Length: {seq_lens.max()}")
        print(f"Average Sequence Length: {avg_seq_len:.2f}")
        print(f"Avg Unique Categories per User: {avg_unique_cats:.2f}")

    print("=" * 40)


def process_and_partition_movielens(csv_file, output_dir="processed_data"):
    """
    Read the generated MovieLens sequential CSV and produce pickle files.

    Expected CSV columns:
        user_id, item_id, rating, timestamp, category

    Output pickle structure:
        [Item sequence, User ID, Category sequence, Target item, Time sequence, Rating sequence]
    """
    print(f"--- Start partitioning file: {csv_file} ---")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    user2id = {}
    item2id = {}
    cat2id = {}

    user_interactions = defaultdict(list)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"Detected header: {header}")

        header_map = {name: idx for idx, name in enumerate(header)}
        required_columns = ["user_id", "item_id", "rating", "timestamp", "category"]

        missing = [col for col in required_columns if col not in header_map]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        count = 0
        for row in reader:
            if len(row) < len(header):
                continue

            u_origin = row[header_map["user_id"]]
            i_origin = row[header_map["item_id"]]
            t_origin = row[header_map["timestamp"]]
            c_origin = row[header_map["category"]]

            try:
                r_origin = float(row[header_map["rating"]])
            except ValueError:
                r_origin = 0.0

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

            count += 1
            if count % 1_000_000 == 0:
                print(f"Processed {count} rows...")

    print(f"Reading completed. Total interactions: {count}")
    print(f"Users: {len(user2id)}, Items: {len(item2id)}, Categories: {len(cat2id)}")

    print("Sorting by timestamp and splitting datasets...")

    data_store = {
        "train": {},
        "valid": {},
        "test": {},
    }

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

    dump_to_pickle(data_store, output_dir)
    save_maps(user2id, item2id, cat2id, output_dir)


def dump_to_pickle(data_store, output_dir):
    """Dump train, valid, and test split files in the six list format."""
    print("Generating Pickle files: [Inputs, User, Cats, Targets, Times, Ratings]")
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

        print(f"Saved {mode}.txt | Number of samples: {len(data_list[0])}")


def save_maps(user2id, item2id, cat2id, output_dir):
    """Save ID mappings and compact statistics."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "id_map.pickle"), "wb") as f:
        pickle.dump(
            {"user2id": user2id, "item2id": item2id, "cat2id": cat2id},
            f,
        )

    with open(os.path.join(output_dir, "stats.txt"), "w", encoding="utf-8") as f:
        f.write(f"num_users={len(user2id)}\n")
        f.write(f"num_items={len(item2id)}\n")
        f.write(f"num_cats={len(cat2id)}\n")

    print("ID mapping tables and statistics have been saved.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified MovieLens 1M preprocessing and partitioning script."
    )

    parser.add_argument("--movies-file", default="movies.dat", help="Path to movies.dat")
    parser.add_argument("--ratings-file", default="ratings.dat", help="Path to ratings.dat")
    parser.add_argument(
        "--intermediate-csv",
        default="movielens_sequential_dataset.csv",
        help="Path to save or read the sequential CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_data",
        help="Directory for train, valid, test, and mapping files",
    )
    parser.add_argument(
        "--min-item-count",
        type=int,
        default=5,
        help="Minimum number of interactions required for an item",
    )
    parser.add_argument(
        "--min-seq-len",
        type=int,
        default=20,
        help="Minimum sequence length required for a user",
    )
    parser.add_argument(
        "--skip-raw-build",
        action="store_true",
        help="Skip raw MovieLens processing and only partition the existing CSV",
    )
    parser.add_argument(
        "--include-title",
        action="store_true",
        help="Also save a title enriched CSV",
    )
    parser.add_argument(
        "--title-output-file",
        default="movielens_sequential_dataset_with_title.csv",
        help="Output path for the optional title enriched CSV",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not args.skip_raw_build:
        build_movielens_sequential_csv(
            movies_file=args.movies_file,
            ratings_file=args.ratings_file,
            output_file=args.intermediate_csv,
            min_item_count=args.min_item_count,
            min_seq_len=args.min_seq_len,
            include_title=args.include_title,
            title_output_file=args.title_output_file,
        )
    else:
        print("Skipping raw MovieLens CSV build.")

    process_and_partition_movielens(
        csv_file=args.intermediate_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
