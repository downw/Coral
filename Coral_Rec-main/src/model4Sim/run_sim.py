import os
import random
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import pipeline
from collections import Counter
import re
from model import SASRec, trans_to_cuda, trans_to_cpu
from utils import Data
from coral import CoralAgent

# Global Config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_USERS = 50
TIME_STEPS = 100  
TOP_K = 5
MODEL_ID = "google/gemma-3-12b-it" 
BATCH_SIZE_LLM = 32
SEED = 2026
DATASET_NAME = 'ML1m'
DATA_DIR = f'../../datasets/{DATASET_NAME}/processed_data/'
CHECKPOINT_DIR = 'checkpoints'
MOVIES_FILE = f'../../datasets/{DATASET_NAME}/movielens_sequential_dataset_with_title.csv'

class ItemMapper:
    def __init__(self, item_to_cat_id_map, movies_file=None):
        self.item_to_cat_id = item_to_cat_id_map
        self.item_text_map = {}
        # Map category IDs to generic names if metadata is missing
        self.fallback_cat_names = {i: f"Category_{i}" for i in set(item_to_cat_id_map.values())}
        
        # Load metadata if available
        if movies_file and os.path.exists(movies_file):
            print(f"Loading metadata from {movies_file}...")
            try:
                df = pd.read_csv(movies_file)
                cols = {c.lower(): c for c in df.columns}
                req_cols = ['item_id', 'title', 'category']
                if all(r in cols for r in req_cols):
                    meta_df = df[[cols['item_id'], cols['title'], cols['category']]].drop_duplicates(subset=[cols['item_id']])
                    for _, row in meta_df.iterrows():
                        iid = int(row[cols['item_id']])
                        title = str(row[cols['title']]).strip()
                        cat = str(row[cols['category']]).strip()
                        self.item_text_map[iid] = f"{title} ({cat})"
                        
                        # Store category name mapping if available
                        cat_id = self.item_to_cat_id.get(iid)
                        if cat_id is not None:
                            self.fallback_cat_names[cat_id] = cat
            except Exception as e:
                print(f"[Error] Failed to load metadata: {e}")

    def get_item_text(self, item_id):
        if item_id in self.item_text_map:
            return self.item_text_map[item_id]
        cat_id = self.item_to_cat_id.get(item_id, -1)
        cat_name = self.fallback_cat_names.get(cat_id, f"Unknown_Cat")
        return f"Item_{item_id} (Type: {cat_name})"

    def get_cat_name(self, cat_id):
        return self.fallback_cat_names.get(cat_id, f"Category_{cat_id}")

def build_user_prompt_aligned(history_texts, rec_texts, preferred_cats_str, domain="movie", personality="bored"):
    """
    Constructs the prompt strictly based on the provided LaTeX Standard Evaluation Prompt.
    """
    # 1. Format History (simplify to avoid token overflow if necessary)
    # Using the last 10 items to represent history context
    recent_history = history_texts[-10:] if len(history_texts) > 10 else history_texts
    history_str = ", ".join([f"'{t}'" for t in recent_history])
    
    # 2. Format Recommendations
    # Format: [1] Title (Cat), [2] Title (Cat)...
    rec_str = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(rec_texts)])
    
    # 3. Define Personality Description based on LaTeX options
    if personality == "obsessive":
        personality_desc = "have obsessive personality"
    else:
        personality_desc = "get bored very quickly"

    # 4. Construct the prompt matching the LaTeX box structure
    prompt_content = (
        f"Role: {domain} User simulator.\n\n"
        
        f"Context: You have a history of [{history_str}] and prefer categories [{preferred_cats_str}]. "
        f"You [{personality_desc}].\n\n"
        
        f"Task: Evaluate the recommendation list:\n{rec_str}\n"
        f"Select its ID (1-{len(rec_texts)}) if you are interested.\n"
        f"Select 0 if you get bored.\n\n" # Unified to 0 to match the 'Output' instruction below
        
        f"Output: choose integer decision only (0 or Item_ID)."
    )
    
    return [{"role": "user", "content": prompt_content}]

def get_recommendations_batch(model, current_seqs, k=5):
    """ Generates SASRec recommendations """
    model.eval()
    max_len = model.max_len
    padded_input = []
    for seq in current_seqs:
        s = seq[-max_len:]
        pad_len = max_len - len(s)
        padded_input.append(s + [0] * pad_len if pad_len > 0 else s)
    
    input_tensor = trans_to_cuda(torch.tensor(padded_input, dtype=torch.long))
    with torch.no_grad():
        hidden = model(input_tensor, attention_mask=None)
        last_hidden = hidden[:, -1, :] 
        scores = model.compute_scores(last_hidden)
        scores = trans_to_cpu(scores).detach().numpy()
        scores[:, 0] = -np.inf # Mask padding item
    return np.argsort(-scores, axis=1)[:, :k]

def simulate_step_llm(llm_pipe, user_prompts):
    """ Calls the LLM and parses the integer output """
    outputs = llm_pipe(
        user_prompts, 
        batch_size=BATCH_SIZE_LLM, 
        max_new_tokens=10, 
        do_sample=True,
        temperature=0.6 
    )
    choices = []
    for out in outputs:
        content = out[0]['generated_text'] if isinstance(out, list) else out['generated_text']
        if isinstance(content, list): content = content[-1]['content']
        text = str(content)
        
        # Parse the last number found in the response
        nums = re.findall(r'\b\d+\b', text) 
        if nums:
            # We look for valid IDs (0 to 5 usually)
            choice = int(nums[-1])
            if 0 <= choice <= TOP_K:
                choices.append(choice)
            else:
                choices.append(0) # Fallback if number is out of range
        else:
            choices.append(0) # Fallback if no number found
    return choices

def apply_coral_reranking(agent, user_ids, base_recs, current_seqs, item_to_cat):
    """ Applies CORAL agent logic to re-rank items based on Hawkes Process """
    reranked_batch = []
    
    for idx, uid in enumerate(user_ids):
        current_t = len(current_seqs[idx])
        # Get optimal category scores from Hawkes process
        _, D_t, cat_scores = agent.get_policy_target_category(
            uid, current_t, strategy='argmax', return_scores=True
        )
        sorted_cats = np.argsort(-cat_scores)
        
        candidates = base_recs[idx]
        cat_buckets = {}
        for item in candidates:
            c = item_to_cat.get(item, 0)
            if c not in cat_buckets:
                cat_buckets[c] = []
            cat_buckets[c].append(item)
            
        new_order = []
        # Re-order candidates based on category priority
        for c in sorted_cats:
            if c in cat_buckets:
                new_order.extend(cat_buckets[c])
                del cat_buckets[c]
        
        # Append remaining items
        for rem in cat_buckets.values():
            new_order.extend(rem)
        
        reranked_batch.append(new_order)
    
    return np.array(reranked_batch)

def calculate_next_intensity_simple(current_intensity, mu, alpha, beta, chosen_cat_id):
    """ Calculates Hawkes Process intensity for the next step """
    decay_factor = np.exp(-beta) 
    decayed = (current_intensity - mu) * decay_factor + mu
    if len(alpha.shape) > 1:
        excitation = alpha[:, chosen_cat_id] 
    else:
        excitation = alpha[chosen_cat_id]
    return decayed + excitation

def get_preferred_categories(cat_seq, mapper, top_n=3):
    """ Helper to get string representation of top categories for the Prompt """
    if not cat_seq:
        return "General"
    counts = Counter(cat_seq)
    top_cats_ids = [c for c, _ in counts.most_common(top_n)]
    top_cats_names = [mapper.get_cat_name(c) for c in top_cats_ids]
    return ", ".join(top_cats_names)

def main():
    # --- Arguments & Setup ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ML1m')
    # ... (Keep existing hyperparams)
    parser.add_argument('--tau', type=float, default=3.5)
    # Add simple args for model config to avoid errors if not passed
    parser.add_argument('--hiddenSize', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_dc', type=float, default=0.1)
    parser.add_argument('--lr_dc_step', type=int, default=10)
    parser.add_argument('--l2', type=float, default=1e-5)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--dropout_local', type=float, default=0.2)
    parser.add_argument('--lambda_max', type=float, default=0.7)
    parser.add_argument('--kappa', type=float, default=1)
    parser.add_argument('--rho', type=float, default=2)
    parser.add_argument('--delta_conf', type=float, default=0.1)
    
    opt = parser.parse_args()
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.set_device(0) 
    
    # --- Load Data ---
    print(f"Loading data from {DATA_DIR}...")
    try:
        with open(os.path.join(DATA_DIR, 'train.txt'), 'rb') as f:
            train_data_raw = pickle.load(f)
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    item_to_cat = {}
    seqs, _, cats, _, _, _ = train_data_raw
    all_items = set()
    for s, c in zip(seqs, cats):
        for item_id, cat_id in zip(s, c):
            if item_id != 0:
                item_to_cat[item_id] = cat_id
                all_items.add(item_id)
    
    num_node = max(all_items)
    num_cats = max(item_to_cat.values()) + 1
    mapper = ItemMapper(item_to_cat, movies_file=MOVIES_FILE)
    
    # --- Load Models ---
    print("Loading SASRec...")
    model = SASRec(opt, num_node)
    model_path = os.path.join(CHECKPOINT_DIR, f'sasrec_best_model_{opt.dataset}.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu')) # Added safe load
        model = trans_to_cuda(model)
    else:
        print(f"Warning: Checkpoint {model_path} not found. Using random init.")
        model = trans_to_cuda(model) # Proceed for testing code flow
    
    print("Loading CORAL Agent...")
    agent = CoralAgent(len(seqs), num_cats, item_to_cat, opt)
    hawkes_path = os.path.join(CHECKPOINT_DIR, f'hawkes_params_{opt.dataset}_tau{opt.tau}.pth')
    if os.path.exists(hawkes_path):
        agent.load_params(hawkes_path)
    
    print(f"Loading LLM: {MODEL_ID}...")
    try:
        dtype_to_use = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        llm = pipeline("text-generation", model=MODEL_ID, device_map="auto", torch_dtype=dtype_to_use)
    except Exception as e:
        print(f"LLM Load Failed (likely VRAM/Path): {e}. Simulation cannot proceed.")
        return

    # --- Initialize Simulation ---
    print(f"Initializing {NUM_USERS} users...")
    start_items = random.sample(list(all_items), NUM_USERS)
    
    sim_users = []
    sas_intensities = {} 

    for i in range(NUM_USERS):
        start_item = start_items[i]
        start_cat = item_to_cat.get(start_item, 0)
        
        sim_users.append({
            'uid': i,
            'sas_seq': [start_item],
            'coral_seq': [start_item],
            'sas_cat_seq': [start_cat],
            'coral_cat_seq': [start_cat],
        })
        
        # Init Intensity
        init_intensity = agent.mu[i] + agent.alpha[i, :, start_cat]
        sas_intensities[i] = init_intensity
        
        # Init CORAL Agent State
        agent.reconstruct_history(i, [start_item], [start_cat], [1.0]) 

    last_action_status_sas = [False] * NUM_USERS # Track if last was skipped (optional for future use)
    sas_d_curve = []
    coral_d_curve = []
    
    domain_dic = {"ML1m": "Movie", "Steam": "Game", "Amazon": "Shopping"}
    current_domain = domain_dic.get(DATASET_NAME, "Movie")

    print("\n=== Starting Simulation (Tracking Intensity D with Exposure Effect) ===")
    
    for t in tqdm(range(TIME_STEPS), desc="Simulating"):
        
        # ==========================
        # PHASE A: SASRec (Shadow Tracking)
        # ==========================
        curr_seqs = [u['sas_seq'] for u in sim_users]
        recs_idx = get_recommendations_batch(model, curr_seqs, k=TOP_K)
        
        prompts = []
        for i, r_list in enumerate(recs_idx):
            # Extract Textual Data
            hist_text = [mapper.get_item_text(x) for x in curr_seqs[i]]
            rec_text = [mapper.get_item_text(x) for x in r_list]
            
            # Extract User Preferences (Top Categories)
            pref_cats_str = get_preferred_categories(sim_users[i]['sas_cat_seq'], mapper)
            
            # Build Prompt (Aligned with LaTeX)
            # Personality: Can be "bored" or "obsessive". Defaulting to "bored" as per snippet.
            prompts.append(build_user_prompt_aligned(
                hist_text, rec_text, pref_cats_str, 
                domain=current_domain,
                personality="bored" 
            ))
        
        choices = simulate_step_llm(llm, prompts)
        
        step_sas_d = []
        for i, choice in enumerate(choices):
            uid = sim_users[i]['uid']
            curr_int = sas_intensities[uid]
            mu_val = agent.mu[uid]
            alpha_val = agent.alpha[uid]
            beta_val = agent.beta[uid]
            
            if choice > 0:
                # === User Clicked (Choice is 1-based index) ===
                last_action_status_sas[i] = False
                chosen_idx = choice - 1
                item = recs_idx[i][chosen_idx]
                cat = item_to_cat.get(item, 0)
                
                sim_users[i]['sas_seq'].append(item)
                sim_users[i]['sas_cat_seq'].append(cat)
                
                target_cat = cat 
            else:
                # === User Bored/Rejected (Exposure Effect) ===
                last_action_status_sas[i] = True
                # Assume exposure to the top ranked item
                top1_item = recs_idx[i][0]
                top1_cat = item_to_cat.get(top1_item, 0)
                target_cat = top1_cat

            new_intensity = calculate_next_intensity_simple(
                curr_int, mu_val, alpha_val, beta_val, 
                chosen_cat_id=target_cat
            )
            
            sas_intensities[uid] = new_intensity
            step_sas_d.append(np.mean(new_intensity))
            
        sas_d_curve.append(np.mean(step_sas_d))

        # ==========================
        # PHASE B: CORAL (Active Intervention)
        # ==========================
        c_seqs = [u['coral_seq'] for u in sim_users]
        uids = [u['uid'] for u in sim_users]
        
        base_recs = get_recommendations_batch(model, c_seqs, k=50)
        coral_recs_idx = apply_coral_reranking(agent, uids, base_recs, c_seqs, item_to_cat)
        coral_recs_idx = coral_recs_idx[:, :TOP_K]

        c_prompts = []
        for i, r_list in enumerate(coral_recs_idx):
            hist_text = [mapper.get_item_text(x) for x in c_seqs[i]]
            rec_text = [mapper.get_item_text(x) for x in r_list]
            
            # Extract User Preferences
            pref_cats_str = get_preferred_categories(sim_users[i]['coral_cat_seq'], mapper)
            
            # Build Prompt
            c_prompts.append(build_user_prompt_aligned(
                hist_text, rec_text, pref_cats_str, 
                domain=current_domain,
                personality="bored"
            ))
            
        c_choices = simulate_step_llm(llm, c_prompts)
        
        step_coral_d = []
        for i, c_choice in enumerate(c_choices):
            uid = sim_users[i]['uid']
            
            if c_choice > 0:
                # === Clicked ===
                target_item = coral_recs_idx[i][c_choice - 1]
                
                sim_users[i]['coral_seq'].append(target_item)
                sim_users[i]['coral_cat_seq'].append(item_to_cat.get(target_item, 0))
            else:
                # === Rejected ===
                target_item = coral_recs_idx[i][0] 

            target_cat = item_to_cat.get(target_item, 0)
            # Update Hawkes process inside the agent
            current_d = agent.update_single_step(uid, target_cat, rating=5.0, time_gap=1.0)
            step_coral_d.append(current_d)
            
        coral_d_curve.append(np.mean(step_coral_d))
        
        # --- Logging ---
        if t == 0 or (t + 1) % 10 == 0:
            print(f"\n=== Step {t+1} Summary ===")
            sas_click_rate = np.mean([1 if c > 0 else 0 for c in choices])
            coral_click_rate = np.mean([1 if c > 0 else 0 for c in c_choices])
            
            print(f"SASRec Click Rate: {sas_click_rate:.2%}")
            print(f"CORAL Click Rate:  {coral_click_rate:.2%}")

    print("\nSaving simulation data...")
    # (Saving logic remains same)
    df_metrics = pd.DataFrame({
        'Step': range(1, TIME_STEPS + 1),
        'SASRec_Intensity_D': sas_d_curve,
        'CORAL_Intensity_D': coral_d_curve
    })
    df_metrics.to_csv('simulation_intensity_d.csv', index=False)
    print("Done!")

if __name__ == '__main__':
    main()