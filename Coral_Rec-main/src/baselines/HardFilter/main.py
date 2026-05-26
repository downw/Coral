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
parser.add_argument('--batch_size', type=int, default=32)
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

# Filter Specific Args
parser.add_argument('--sim_threshold', type=float, default=0.7, 
                    help='Hard threshold for similarity. Items with sim > threshold with ANY history item will be removed.')

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
    num_cats = max(item_to_cat.values()) + 1
    return item_to_cat, cat_to_items, num_cats

def forward_train(model, data):
    from model import forward
    return forward(model, data, mode='train')

def analyze_tail_risk_stratified(records, metric_keys, percentiles=[0.20, 0.10, 0.05]):
    if not records: return {}
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
        avg_risk = np.mean([x['risk_val'] for x in subset])
        
        for k in metric_keys:
            vals = [x['metrics'][k] for x in subset]
            summary[f"Top{int(p*100)}%_Risk_{k}"] = np.mean(vals)
        summary[f"Top{int(p*100)}%_AvgHD"] = avg_risk
    return summary

def print_report(records, k_list, title):
    """ Helper to print the stratified analysis table """
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
    """ Helper to calculate metrics for a single recommendation list """
    metrics = {}
    for K in k_list:
        # Note: If filtering reduced list size below K, we take what we have
        topk = topk_list[:K]
        hit = 1.0 if target_item in topk else 0.0
        metrics[f'Recall@{K}'] = hit
        
        try:
            rank = topk.index(target_item)
            metrics[f'MRR@{K}'] = 1.0 / (rank + 1)
        except ValueError:
            metrics[f'MRR@{K}'] = 0.0
        
        # TCC Calculation
        if local_dominant_cat != -1 and len(topk) > 0:
            match_count = 0
            for x in topk:
                xc = item_to_cat.get(x, -1)
                if xc == local_dominant_cat:
                    match_count += 1
            metrics[f'TCC@{K}'] = match_count / len(topk) # Normalize by actual list length
        else:
            metrics[f'TCC@{K}'] = 0.0
    return metrics

def evaluate_hard_filter(model, test_data, item_to_cat, num_cats, sim_threshold, desc="Eval"):
    model.eval()
    loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=opt.batch_size, shuffle=False)
    k_list = [5, 10]
    MIN_HISTORY_LEN = 4 
    
    # [Step A] Prepare Embeddings
    if hasattr(model, 'module'):
        item_emb_weight = model.module.item_embedding.weight.detach().cpu().numpy()
    else:
        item_emb_weight = model.item_embedding.weight.detach().cpu().numpy()
    
    emb_norm = np.linalg.norm(item_emb_weight, axis=1, keepdims=True)
    item_emb_normed = item_emb_weight / (emb_norm + 1e-8)

    records_base = []
    records_filter = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc=desc):
            inputs, mask, targets, cat_inputs, rate_inputs, user_ids = data
            inputs = trans_to_cuda(inputs).long()
            
            hidden = model(inputs, attention_mask=None) 
            all_logits = model.compute_scores(hidden) 
            all_logits = trans_to_cpu(all_logits).detach().numpy()
            
            inputs_np = inputs.cpu().numpy()
            cat_inputs_np = cat_inputs.numpy()
            batch_size = inputs.shape[0]
            
            for b in range(batch_size):
                u_items = inputs_np[b]
                u_cats = cat_inputs_np[b]
                
                valid_idx = np.where(u_items != 0)[0]
                if len(valid_idx) <= MIN_HISTORY_LEN:
                    continue
                
                seq_items = u_items[valid_idx]
                seq_cats = u_cats[valid_idx]
                seq_logits = all_logits[b, valid_idx, :] 
                
                for t in range(MIN_HISTORY_LEN - 1, len(seq_items) - 1):
                    target_item = seq_items[t+1]
                    
                    # [Step B] Risk Quantification (Shared)
                    window_len = 20
                    window_start = max(0, t - window_len + 1)
                    recent_cats = seq_cats[window_start:t+1]
                    valid_recent_cats = [c for c in recent_cats if c < num_cats]
                    
                    hd_val = 0.0
                    local_dominant_cat = -1
                    if valid_recent_cats:
                        counts = np.bincount(valid_recent_cats)
                        local_dominant_cat = np.argmax(counts)
                        hd_val = counts[local_dominant_cat] / len(valid_recent_cats)
                    
                    # [Step C] History Items for Filtering
                    history_items = seq_items[window_start:t+1]
                    hist_embs = item_emb_normed[history_items] # shape: [L, D]

                    # [Step D] Candidate Generation
                    step_scores = seq_logits[t].copy()
                    step_scores[0] = -np.inf 
                    
                    # Get candidates (Top 200) for re-ranking
                    rerank_k = 200
                    ind = np.argpartition(step_scores, -rerank_k)[-rerank_k:]
                    
                    # 1. Base Strategy (No Filter)
                    base_scores = step_scores[ind]
                    sorted_base_idx = np.argsort(-base_scores)
                    final_list_base = ind[sorted_base_idx].tolist()
                    
                    metrics_base = calculate_metrics_for_list(
                        final_list_base, target_item, k_list, item_to_cat, local_dominant_cat
                    )
                    records_base.append({'risk_val': float(hd_val), 'metrics': metrics_base})
                    
                    # 2. Hard Filter Strategy
                    cand_embs = item_emb_normed[ind] # shape: [K, D]
                    
                    # Calculate Similarity Matrix: [K, L]
                    # Sim(Candidate_i, History_j)
                    sim_matrix = np.dot(cand_embs, hist_embs.T)
                    
                    # Find max similarity for each candidate against ANY history item
                    # Shape: [K]
                    max_sim_per_cand = np.max(sim_matrix, axis=1)
                    
                    # Hard Filter: Keep only items where max_sim <= threshold
                    valid_mask = max_sim_per_cand <= sim_threshold
                    valid_indices_local = np.where(valid_mask)[0]
                    
                    if len(valid_indices_local) == 0:
                        # Fallback: If all filtered, return empty list (Recall will be 0)
                        final_list_filter = []
                    else:
                        # Retrieve original indices and scores for valid items
                        valid_global_ind = ind[valid_indices_local]
                        valid_scores = step_scores[valid_global_ind]
                        
                        # Sort by original SASRec score
                        sorted_filter_idx = np.argsort(-valid_scores)
                        final_list_filter = valid_global_ind[sorted_filter_idx].tolist()
                    
                    metrics_filter = calculate_metrics_for_list(
                        final_list_filter, target_item, k_list, item_to_cat, local_dominant_cat
                    )
                    records_filter.append({'risk_val': float(hd_val), 'metrics': metrics_filter})

    # --- Reporting ---
    print_report(records_base, k_list, "BASE MODEL (SASRec)")
    results_filter = print_report(records_filter, k_list, f"HARD FILTER (Thresh={sim_threshold})")
    
    return results_filter.get("Global_Recall@10", 0.0), results_filter.get("Global_MRR@10", 0.0)

def train_sasrec_pipeline(model, train_data, test_data, best_model_path, item_to_cat, num_cats):
    print("\n[Step 1] Start Training SASRec Baseline...")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    
    best_hit = 0.0
    bad_counter = 0
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

        # Validation (Threshold irrelevant here, using 1.0 means no filtering basically)
        hit, mrr = evaluate_hard_filter(
            model, 
            test_data, 
            item_to_cat=item_to_cat, 
            num_cats=num_cats, 
            sim_threshold=1.0, 
            desc="Valid"
        )
        
        if hit > best_hit:
            best_hit = hit
            torch.save(model.state_dict(), best_model_path)
            bad_counter = 0
            print(f"New best model saved! Recall@10: {best_hit:.4f}")
        else:
            bad_counter += 1
            print(f"No improvement. Bad counter: {bad_counter}")
            if bad_counter >= opt.patience:
                print("Early stopping triggered.")
                break

def main():
    setup_logger(opt.dataset, opt)
    redirect_tqdm_to_console()
    init_seed(2020)
    
    # Paths
    train_data_raw = pickle.load(open(os.path.join('../../../../datasets/', opt.dataset, 'processed_data/train.txt'), "rb"))
    test_data_raw = pickle.load(open(os.path.join('../../../../datasets/', opt.dataset, 'processed_data/test.txt'), "rb"))
    
    num_node = max(get_num_node_unique(train_data_raw), get_num_node_unique(test_data_raw))
    item_to_cat, cat_to_items, num_cats = build_maps(train_data_raw)
    
    train_data = Data(train_data_raw, num_node, opt.max_len, opt.mask_prob, mode='train')
    test_data = Data(test_data_raw, num_node, opt.max_len, opt.mask_prob, mode='test')
    
    model = trans_to_cuda(SASRec(opt, num_node))
    save_dir = '../../checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, f'sasrec_best_model_{opt.dataset}.pth')
    
    if not os.path.exists(best_model_path):
        print(f"Pre-trained model not found. Starting training...")
        train_sasrec_pipeline(model, train_data, test_data, best_model_path, item_to_cat, num_cats)
    else:
        print(f"Found pre-trained model. Loading...")
    
    model.load_state_dict(torch.load(best_model_path))

    print("\n=== Evaluation Phase (Hard Filter Baseline) ===")
    print(f"Similarity Threshold: {opt.sim_threshold}")
    
    evaluate_hard_filter(
        model, 
        test_data, 
        item_to_cat=item_to_cat, 
        num_cats=num_cats, 
        sim_threshold=opt.sim_threshold,
        desc="Testing"
    )

if __name__ == '__main__':
    main()