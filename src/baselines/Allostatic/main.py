import time
import argparse
import pickle
import random
import os
import numpy as np
import torch
from tqdm import tqdm
from model import SASRec, trans_to_cuda, trans_to_cpu
from utils import Data, get_num_node_unique
from log import setup_logger, redirect_tqdm_to_console
from allostasis import AllostaticRegulator

def init_seed(seed=None):
    if seed is None: seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if torch.cuda.is_available():
    torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='ML1m')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128) # Increased for eval speed
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_dc', type=float, default=0.1)
parser.add_argument('--lr_dc_step', type=int, default=3)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--dropout_local', type=float, default=0.2)
parser.add_argument('--max_len', type=int, default=200)
parser.add_argument('--mask_prob', type=float, default=0.2)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--valid_portion', type=float, default=0.1)
parser.add_argument('--patience', type=int, default=3)

# Allostatic Params
parser.add_argument('--Ib', type=float, default=0.1, help='Penalty weight')
parser.add_argument('--lambda_b', type=float, default=0.01, help='Decay constant')

opt = parser.parse_args()

def build_maps(dataset_raw):
    item_to_cat = {}
    cat_to_items = {}
    seqs, _, cats, _, _, _ = dataset_raw
    for s, c in zip(seqs, cats):
        for item_id, cat_id in zip(s, c):
            if item_id != 0:
                item_to_cat[item_id] = cat_id
                if cat_id not in cat_to_items:
                    cat_to_items[cat_id] = set()
                cat_to_items[cat_id].add(item_id)
    for c in cat_to_items:
        cat_to_items[c] = list(cat_to_items[c])
    num_cats = max(item_to_cat.values()) + 1 if item_to_cat else 1
    return item_to_cat, cat_to_items, num_cats

def forward_train(model, data):
    from model import forward
    return forward(model, data, mode='train')

# =========================================================
#  Reporting & Analysis Helpers
# =========================================================

def analyze_tail_risk_stratified(records, metric_keys, percentiles=[0.20, 0.10, 0.05]):
    if not records: return {}
    # Sort by Risk (HD) descending
    sorted_records = sorted(records, key=lambda x: x['risk_val'], reverse=True)
    n = len(sorted_records)
    summary = {}
    
    # Global Mean
    for k in metric_keys:
        vals = [r['metrics'][k] for r in sorted_records]
        summary[f"Global_{k}"] = np.mean(vals)
        
    # Percentiles
    for p in percentiles:
        cutoff = int(np.ceil(n * p))
        cutoff = max(1, cutoff)
        subset = sorted_records[:cutoff]
        
        # Calculate avg metrics for this subset
        for k in metric_keys:
            vals = [x['metrics'][k] for x in subset]
            summary[f"Top{int(p*100)}%_Risk_{k}"] = np.mean(vals)
        
        # Also track the Avg HD for the top 5% group explicitly for the footer
        if abs(p - 0.05) < 0.001:
            avg_risk = np.mean([x['risk_val'] for x in subset])
            summary[f"Top{int(p*100)}%_AvgHD"] = avg_risk
            
    return summary

def print_report(records, k_list, title):
    metric_keys = []
    for K in k_list:
        metric_keys.extend([f'Recall@{K}', f'MRR@{K}', f'TCC@{K}'])
        
    results = analyze_tail_risk_stratified(records, metric_keys, percentiles=[0.20, 0.10, 0.05])
    
    print("\n" + "="*25 + f" {title} " + "="*25)
    print(f"Total Contexts: {len(records)}")
    
    headers = ["Metric", "Global Avg", "Top 20% HD", "Top 10% HD", "Top 5% HD"]
    row_fmt = "{:<12} | {:<10.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f}"
    
    print("-" * 75)
    print("{:<12} | {:<10} | {:<12} | {:<12} | {:<12}".format(*headers))
    print("-" * 75)
    
    for m_type in ['Recall', 'MRR', 'TCC']:
        for K in k_list:
            base_key = f"{m_type}@{K}"
            val_global = results.get(f"Global_{base_key}", 0.0)
            val_20 = results.get(f"Top20%_Risk_{base_key}", 0.0)
            val_10 = results.get(f"Top10%_Risk_{base_key}", 0.0)
            val_05 = results.get(f"Top5%_Risk_{base_key}", 0.0)
            print(row_fmt.format(base_key, val_global, val_20, val_10, val_05))
            
    print("-" * 75)
    print(f"Avg HD (Top 5%): {results.get('Top5%_AvgHD', 0):.2f}")
    print("="*75 + "\n")
    return results

def calculate_metrics_for_list(topk_list, target_item, k_list, item_to_cat, local_dominant_cat):
    metrics = {}
    for K in k_list:
        topk = topk_list[:K]
        hit = 1.0 if target_item in topk else 0.0
        metrics[f'Recall@{K}'] = hit
        metrics[f'MRR@{K}'] = 1.0 / (topk.index(target_item) + 1) if hit else 0.0
        
        # TCC: Fraction of topk that matches the user's dominant category
        if local_dominant_cat != -1:
            match_count = 0
            for x in topk:
                xc = item_to_cat.get(x, -1)
                if xc == local_dominant_cat:
                    match_count += 1
            metrics[f'TCC@{K}'] = match_count / K
        else:
            metrics[f'TCC@{K}'] = 0.0
    return metrics

# =========================================================
#  Evaluation Loop
# =========================================================

def evaluate_trajectory(model, test_data, item_to_cat, num_cats, I_b, lambda_b):
    """
    Evaluates the model on the full trajectory of user interactions in the test set.
    Generates report for Base and Calibrated models.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=opt.batch_size, shuffle=False)
    
    regulator = AllostaticRegulator(item_to_cat, num_cats, I_b, lambda_b)
    
    k_list = [5, 10]
    records_base = []
    records_calib = []
    
    MIN_HISTORY = 4 # Skip very short histories
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating Trajectories"):
            inputs, mask, targets, cat_inputs, rate_inputs, user_ids = data
            inputs = trans_to_cuda(inputs).long()
            hidden = model(inputs, attention_mask=None) 
            #  [B, Max_Len, Num_Items]
            all_logits = model.compute_scores(hidden)
            all_logits = trans_to_cpu(all_logits).detach().numpy()
            # Prepare for Python loop over batch
            inputs_np = inputs.cpu().numpy()
            cat_inputs_np = cat_inputs.numpy()
            
            batch_size = inputs.shape[0]
            
            for b in range(batch_size):
                u_seq = inputs_np[b]       # [Item1, Item2, ..., 0, 0]
                u_cats = cat_inputs_np[b]
                valid_idx = np.where(u_seq != 0)[0]
                for i in range(len(valid_idx) - 1):
                    t = valid_idx[i]     # Index of last item in history
                    target_idx = valid_idx[i+1] # Index of target item
                    # History length check
                    current_hist_len = i + 1
                    if current_hist_len < MIN_HISTORY:
                        continue
                        
                    target_item = u_seq[target_idx]
                    hist_cats = u_cats[valid_idx[:i+1]]
                    hist_items = u_seq[valid_idx[:i+1]]
                    dom_cat, hd_val = regulator.get_dominant_cat_and_hd(hist_cats)
                    step_logits = all_logits[b, t, :].copy()
                    step_logits[0] = -np.inf # Mask padding
                    rerank_k = 50 
                    ind = np.argpartition(step_logits, -rerank_k)[-rerank_k:]
                    top_scores = step_logits[ind]
                    sorted_idx = np.argsort(-top_scores)
                    final_base = ind[sorted_idx].tolist()
                    m_base = calculate_metrics_for_list(final_base, target_item, k_list, item_to_cat, dom_cat)
                    records_base.append({'risk_val': float(hd_val), 'metrics': m_base})
                    penalty = regulator.get_step_penalty(hist_items, hist_cats, i).cpu().numpy()
                    calib_logits = step_logits - penalty
                    ind_c = np.argpartition(calib_logits, -rerank_k)[-rerank_k:]
                    top_scores_c = calib_logits[ind_c]
                    sorted_idx_c = np.argsort(-top_scores_c)
                    final_calib = ind_c[sorted_idx_c].tolist()
                    m_calib = calculate_metrics_for_list(final_calib, target_item, k_list, item_to_cat, dom_cat)
                    records_calib.append({'risk_val': float(hd_val), 'metrics': m_calib})

    print_report(records_base, k_list, "BASE MODEL (SASRec)")
    results_calib = print_report(records_calib, k_list, f"CALIBRATED (Lambda={lambda_b})")
    return results_calib.get("Global_Recall@10", 0.0)

def train_sasrec(model, train_data, test_data, best_model_path, item_to_cat, num_cats):
    print("\n[Step 1] Start Training SASRec Baseline...")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    
    best_hit = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    
    for epoch in range(opt.epoch):
        model.train()
        total_loss = 0.0
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            targets, scores = forward_train(model, data)
            loss = model.loss_function(scores, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch} loss: {total_loss:.4f}")
        torch.save(model.state_dict(), best_model_path) 

def main():
    setup_logger(opt.dataset, opt)
    redirect_tqdm_to_console()
    init_seed(2020)
    
    # Paths
    path_train = os.path.join('../../../../datasets/', opt.dataset, 'processed_data/train.txt')
    path_test = os.path.join('../../../../datasets/', opt.dataset, 'processed_data/test.txt')
    train_data_raw = pickle.load(open(path_train, "rb"))
    test_data_raw = pickle.load(open(path_test, "rb"))
    
    num_node = max(get_num_node_unique(train_data_raw), get_num_node_unique(test_data_raw))
    item_to_cat, cat_to_items, num_cats = build_maps(train_data_raw)
    train_data = Data(train_data_raw, num_node, opt.max_len, opt.mask_prob, mode='train')
    test_data = Data(test_data_raw, num_node, opt.max_len, opt.mask_prob, mode='test')
    model = trans_to_cuda(SASRec(opt, num_node))
    save_dir = '../../checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f'sasrec_best_model_{opt.dataset}.pth')
    
    if not os.path.exists(best_model_path):
        train_sasrec(model, train_data, test_data, best_model_path, item_to_cat, num_cats)
    else:
        print(f"Found pre-trained model at {best_model_path}. Loading...")
        model.load_state_dict(torch.load(best_model_path))
    print("\n=== Evaluation Phase: Stratified Risk Analysis ===")
    evaluate_trajectory(
        model, 
        test_data, 
        item_to_cat=item_to_cat, 
        num_cats=num_cats, 
        I_b=opt.Ib,
        lambda_b=opt.lambda_b
    )

if __name__ == '__main__':
    main()