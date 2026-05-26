import csv
import pickle
import os
from collections import defaultdict

def process_and_partition_steam(csv_file, output_dir="processed_data"):
    print(f"--- Start processing file: {csv_file} ---")
    
    # 1. Initialize ID mapping dictionaries
    user2id = {} 
    item2id = {} 
    cat2id  = {} 
    
    # Storage structure: UserID -> List of (Timestamp, ItemID, CatID, Hours)
    # Note: Hours is stored at the end, replacing the previous Rating position
    User_interactions = defaultdict(list)
    
    # 2. Read CSV directly
    # Header format: user_id, item_id, timestamp, category, hours
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            print(f"Detected header: {header}")
            
            count = 0
            for row in reader:
                # Parse according to the new header:
                # user_id, item_id, timestamp, category, hours
                # row[0]: user_id
                # row[1]: item_id
                # row[2]: timestamp
                # row[3]: category
                # row[4]: hours
                
                if len(row) < 5:
                    continue

                u_origin = row[0]
                i_origin = row[1]
                t_origin = row[2]
                c_origin = row[3]
                
                try:
                    h_origin = float(row[4])  # Read Hours and convert to float
                except ValueError:
                    h_origin = 0.0  # Handle invalid values
                
                # --- ID mapping logic ---
                if u_origin not in user2id:
                    user2id[u_origin] = len(user2id) + 1
                if i_origin not in item2id:
                    item2id[i_origin] = len(item2id) + 1
                if c_origin not in cat2id:
                    cat2id[c_origin] = len(cat2id) + 1
                
                u_int = user2id[u_origin]
                i_int = item2id[i_origin]
                c_int = cat2id[c_origin]
                
                # Store tuple: (timestamp, item, category, hours)
                User_interactions[u_int].append((t_origin, i_int, c_int, h_origin))
                
                count += 1
                if count % 1_000_000 == 0:
                    print(f"Processed {count} rows...")
                    
        print(f"Finished reading. Total interactions: {count}")
        print(f"Users: {len(user2id)}, Items: {len(item2id)}, Categories: {len(cat2id)}")

    except FileNotFoundError:
        print(f"Error: File not found {csv_file}")
        return

    # 3. Sort and split (Train / Valid / Test)
    print("Sorting by timestamp and splitting datasets...")
    
    # Temporary storage for split results
    data_store = {
        'train': {},
        'valid': {},
        'test': {}
    }
    
    for u_int, interactions in User_interactions.items():
        # Sort by timestamp (x[0])
        interactions.sort(key=lambda x: x[0])
        
        # Unpack data (now including hours)
        seq_times, seq_items, seq_cats, seq_hours = zip(*interactions)
        
        seq_times = list(seq_times)
        seq_items = list(seq_items)
        seq_cats  = list(seq_cats)
        seq_hours = list(seq_hours)  # Hours sequence
        
        n = len(seq_items)
        
        # --- Splitting logic ---
        if n < 3:
            # Too few interactions: put all into training set
            data_store['train'][u_int] = (seq_items, seq_cats, seq_times, seq_hours)
            data_store['valid'][u_int] = ([], [], [], [])
            data_store['test'][u_int]  = ([], [], [], [])
        else:
            # Train: remove last two interactions
            data_store['train'][u_int] = (
                seq_items[:-2],
                seq_cats[:-2],
                seq_times[:-2],
                seq_hours[:-2]
            )
            
            # Valid: second last interaction as target
            data_store['valid'][u_int] = (
                seq_items[-2],
                seq_cats[-2],
                seq_times[-2],
                seq_hours[-2]
            )
            
            # Test: last interaction as target
            data_store['test'][u_int] = (
                seq_items[-1],
                seq_cats[-1],
                seq_times[-1],
                seq_hours[-1]
            )

    # 4. Generate Pickle files
    dump_to_pickle(data_store, output_dir)
    
    # Save ID mappings
    save_maps(user2id, item2id, cat2id, output_dir)


def dump_to_pickle(data_store, output_dir):
    # Updated description: the Rating position now stores Hours
    print("Generating Pickle files (structure: [Inputs, User, Cats, Targets, Times, Hours])...")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize data structures with 6 lists
    # 0: Item sequence
    # 1: UserID
    # 2: Category sequence
    # 3: Target item
    # 4: Time sequence
    # 5: Hours sequence (formerly Rating)
    datasets = {
        'train': [[], [], [], [], [], []],
        'valid': [[], [], [], [], [], []],
        'test':  [[], [], [], [], [], []]
    }

    # Ensure consistent user order
    users = sorted(data_store['train'].keys())

    for u in users:
        # Unpack Train (sequence)
        train_i, train_c, train_t, train_h = data_store['train'][u]
        # Unpack Valid (single item)
        valid_i, valid_c, valid_t, valid_h = data_store['valid'][u]
        # Unpack Test (single item)
        test_i,  test_c,  test_t,  test_h  = data_store['test'][u]
        
        # Check whether valid/test data exists
        has_valid = valid_i != [] and isinstance(valid_i, int)
        has_test  = test_i  != [] and isinstance(test_i, int)

        # --- Fill Train ---
        datasets['train'][0].append(train_i)
        datasets['train'][1].append([u])
        datasets['train'][2].append(train_c)
        datasets['train'][3].append(valid_i if has_valid else 0)
        datasets['train'][4].append(train_t)
        datasets['train'][5].append(train_h)  # Hours sequence for training input

        # --- Fill Valid ---
        if has_valid:
            datasets['valid'][0].append(train_i)   # Historical input
            datasets['valid'][1].append([u])
            datasets['valid'][2].append(train_c)   # Category history
            datasets['valid'][3].append(valid_i)   # Target item
            datasets['valid'][4].append(train_t)
            datasets['valid'][5].append(train_h)   # Hours input sequence

        # --- Fill Test ---
        if has_test:
            # Test input usually includes the valid item
            if has_valid:
                test_seq_i = train_i + [valid_i]
                test_seq_c = train_c + [valid_c]
                test_seq_t = train_t + [valid_t]
                test_seq_h = train_h + [valid_h]  # Concatenate hours
            else:
                test_seq_i = train_i
                test_seq_c = train_c
                test_seq_t = train_t
                test_seq_h = train_h
            
            datasets['test'][0].append(test_seq_i)
            datasets['test'][1].append([u])
            datasets['test'][2].append(test_seq_c)
            datasets['test'][3].append(test_i)     # Target item
            datasets['test'][4].append(test_seq_t)
            datasets['test'][5].append(test_seq_h) # Hours input sequence

    # Write to disk
    for mode, data_list in datasets.items():
        path = os.path.join(output_dir, f"{mode}.txt")
        with open(path, 'wb') as f:
            pickle.dump(data_list, f)
        
        # Print statistics
        print(f"Saved {mode}.txt | Number of samples: {len(data_list[0])}")


def save_maps(user2id, item2id, cat2id, output_dir):
    with open(os.path.join(output_dir, 'id_map.pickle'), 'wb') as f:
        pickle.dump(
            {'user2id': user2id, 'item2id': item2id, 'cat2id': cat2id},
            f
        )
    print("ID mapping tables have been saved.")


if __name__ == "__main__":
    # Input file
    input_file = 'steam_sequential_mapped.csv'
    
    if os.path.exists(input_file):
        process_and_partition_steam(input_file)
    else:
        print(f"Error: {input_file} not found in the current directory")
