import csv
import pickle
import os
from collections import defaultdict

def process_and_partition_amazon(csv_file, output_dir="processed_data", k_core_threshold=5):
    print(f"--- Start processing Amazon dataset: {csv_file} ---")
    print(f"--- Filtering threshold (k-core): {k_core_threshold} ---")

    # 1. First pass: load raw data and count frequencies (for k-core filtering)
    # New Amazon CSV header: user_id, item_id, timestamp, category, rating

    raw_interactions = []  # temporary storage of (u_origin, i_origin, t_origin, c_origin, r_origin)
    user_count = defaultdict(int)
    item_count = defaultdict(int)

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            print(f"Header found: {header}")

            count = 0
            for row in reader:
                # Ensure the row is complete
                if len(row) < 5:
                    continue

                # row[0]: user_id
                # row[1]: item_id
                # row[2]: timestamp
                # row[3]: category
                # row[4]: rating

                u_origin = row[0]
                i_origin = row[1]
                t_origin = row[2]
                c_origin = row[3]

                try:
                    r_origin = float(row[4])  # read rating
                except ValueError:
                    r_origin = 0.0  # exception handling

                user_count[u_origin] += 1
                item_count[i_origin] += 1

                # store 5-tuple
                raw_interactions.append((u_origin, i_origin, t_origin, c_origin, r_origin))

                count += 1
                if count % 1_000_000 == 0:
                    print(f"Read {count} lines...")

        print(f"Raw data loading finished. Total lines: {count}")

    except FileNotFoundError:
        print(f"Error: file not found {csv_file}")
        return

    # 2. Build ID mappings (apply k-core filtering at the same time)
    print("Building ID mappings and applying filtering...")

    user2id = {}
    item2id = {}
    cat2id  = {}

    # Select users and items that satisfy the threshold
    valid_users = {u for u, c in user_count.items() if c >= k_core_threshold}
    valid_items = {i for i, c in item_count.items() if c >= k_core_threshold}

    print(f"Before filtering - Users: {len(user_count)}, Items: {len(item_count)}")
    print(f"After filtering  - Users: {len(valid_users)}, Items: {len(valid_items)}")

    # 3. Data transformation (Filtering & Mapping & Grouping)
    # Storage format: UserID -> List of (Timestamp, ItemID, CatID, Rating)
    User_interactions = defaultdict(list)
    skipped_count = 0

    for u_origin, i_origin, t_origin, c_origin, r_origin in raw_interactions:
        # Filtering check
        if u_origin not in valid_users or i_origin not in valid_items:
            skipped_count += 1
            continue

        # ID mapping (start from 1)
        if u_origin not in user2id: user2id[u_origin] = len(user2id) + 1
        if i_origin not in item2id: item2id[i_origin] = len(item2id) + 1
        if c_origin not in cat2id:  cat2id[c_origin]  = len(cat2id) + 1

        u_int = user2id[u_origin]
        i_int = item2id[i_origin]
        c_int = cat2id[c_origin]
        # r_origin is already float

        # Store: (time, item, category, rating)
        User_interactions[u_int].append((t_origin, i_int, c_int, r_origin))

    print(f"Transformation finished. Skipped (filtered) interactions: {skipped_count}")
    print(f"Final valid interactions: {len(raw_interactions) - skipped_count}")

    # 4. Sorting and splitting (Train / Valid / Test)
    print("Sorting by timestamp and splitting datasets...")

    data_store = {
        'train': {},
        'valid': {},
        'test': {}
    }

    for u_int, interactions in User_interactions.items():
        # Sort by timestamp
        interactions.sort(key=lambda x: x[0])

        # Unpack data (now including real ratings)
        seq_times, seq_items, seq_cats, seq_ratings = zip(*interactions)

        seq_times   = list(seq_times)
        seq_items   = list(seq_items)
        seq_cats    = list(seq_cats)
        seq_ratings = list(seq_ratings)

        n = len(seq_items)

        # Leave-one-out split
        if n < 3:
            data_store['train'][u_int] = (seq_items, seq_cats, seq_times, seq_ratings)
            data_store['valid'][u_int] = ([], [], [], [])
            data_store['test'][u_int]  = ([], [], [], [])
        else:
            data_store['train'][u_int] = (seq_items[:-2], seq_cats[:-2], seq_times[:-2], seq_ratings[:-2])
            data_store['valid'][u_int] = (seq_items[-2],  seq_cats[-2],  seq_times[-2],  seq_ratings[-2])
            data_store['test'][u_int]  = (seq_items[-1],  seq_cats[-1],  seq_times[-1],  seq_ratings[-1])

    # 5. Print statistics
    num_users = len(user2id)
    num_items = len(item2id)
    total_inter = len(raw_interactions) - skipped_count

    if num_users > 0 and num_items > 0:
        sparsity = 1 - (total_inter / (num_users * num_items))
        avg_len = total_inter / num_users
        print("-" * 30)
        print("### Dataset Statistics ###")
        print(f"User Count: {num_users}")
        print(f"Item Count: {num_items}")
        print(f"Total Interactions: {total_inter}")
        print(f"Avg Sequence Length: {avg_len:.2f}")
        print(f"Sparsity: {sparsity:.6f} ({sparsity*100:.4f}%)")
        print("-" * 30)

    # 6. Generate Pickle files
    dump_to_pickle(data_store, output_dir)
    save_maps(user2id, item2id, cat2id, output_dir)


def dump_to_pickle(data_store, output_dir):
    print("Generating Pickle files (Unified Format: [Item, User, Cat, Target, Time, Rating])...")
    os.makedirs(output_dir, exist_ok=True)

    # 6 lists:
    # Index 0: Item sequence
    # Index 1: User ID
    # Index 2: Category sequence
    # Index 3: Target item
    # Index 4: Time sequence
    # Index 5: Rating sequence (real ratings)
    datasets = {
        'train': [[], [], [], [], [], []],
        'valid': [[], [], [], [], [], []],
        'test':  [[], [], [], [], [], []]
    }

    users = sorted(data_store['train'].keys())

    for u in users:
        train_i, train_c, train_t, train_r = data_store['train'][u]
        valid_i, valid_c, valid_t, valid_r = data_store['valid'][u]
        test_i,  test_c,  test_t,  test_r  = data_store['test'][u]

        has_valid = valid_i != [] and isinstance(valid_i, int)
        has_test  = test_i  != [] and isinstance(test_i, int)

        # Train
        datasets['train'][0].append(train_i)
        datasets['train'][1].append([u])
        datasets['train'][2].append(train_c)
        datasets['train'][3].append(valid_i if has_valid else 0)
        datasets['train'][4].append(train_t)
        datasets['train'][5].append(train_r)

        # Validation
        if has_valid:
            datasets['valid'][0].append(train_i)
            datasets['valid'][1].append([u])
            datasets['valid'][2].append(train_c)
            datasets['valid'][3].append(valid_i)
            datasets['valid'][4].append(train_t)
            datasets['valid'][5].append(train_r)

        # Test
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

            datasets['test'][0].append(test_seq_i)
            datasets['test'][1].append([u])
            datasets['test'][2].append(test_seq_c)
            datasets['test'][3].append(test_i)
            datasets['test'][4].append(test_seq_t)
            datasets['test'][5].append(test_seq_r)

    for mode, data_list in datasets.items():
        path = os.path.join(output_dir, f"{mode}.txt")
        with open(path, 'wb') as f:
            pickle.dump(data_list, f)
        print(f"Saved {mode}.txt | Number of samples: {len(data_list[0])}")


def save_maps(user2id, item2id, cat2id, output_dir):
    with open(os.path.join(output_dir, 'id_map.pickle'), 'wb') as f:
        pickle.dump({'user2id': user2id, 'item2id': item2id, 'cat2id': cat2id}, f)

    # Also save statistics
    with open(os.path.join(output_dir, 'stats.txt'), 'w') as f:
        f.write(f"num_users={len(user2id)}\n")
        f.write(f"num_items={len(item2id)}\n")
        f.write(f"num_cats={len(cat2id)}\n")

    print("ID mappings and statistics files have been saved.")


if __name__ == "__main__":
    input_file = 'amazon_recent_20m_dataset.csv'

    if os.path.exists(input_file):
        # Default k-core threshold = 5
        process_and_partition_amazon(input_file, k_core_threshold=5)
    else:
        print(f"Error: {input_file} not found in the current directory")
