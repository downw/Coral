import time
import argparse
import pickle
import random
import os
import numpy as np
import torch
from tqdm import tqdm
from model import SASRec, trans_to_cuda, trans_to_cpu
from utils import Data, get_num_node_unique, RGRecManager
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
parser.add_argument('--patience', type=int, default=3)

# RGRec Specific Args
parser.add_argument('--rgrec_weight', type=float, default=0.2, 
                    help='Weight parameter w to control the proportion of GI (Nudge Items)')

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
    # Sort by HD (Risk) value descending
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
    """ Generate the specific table format requested """
    metric_keys = []
    for K in k_list:
        metric_keys.extend([f'Recall@{K}', f'MRR@{K}', f'TCC@{K}'])
        
    results = analyze_tail_risk_stratified(records, metric_keys, percentiles=[0.20, 0.10, 0.05])
    
    print("\n" + "="*25 + f" {title} " + "="*25)
    print(f"Total Contexts: {len(records)}")
    
    headers = ["Metric", "Global Avg", "Top 20% HD", "Top 10% HD", "Top 5% HD"]
    # Adjust spacing to match target output
    row_fmt = "{:<12} | {:<10.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f}"
    
    print("-" * 75)
    print("{:<12} | {:<10} | {:<12} | {:<12} | {:<12}".format(*headers))
    print("-" * 75)
    
    for K in k_list:
        # Print metrics in groups of K
        for m_type in ['Recall', 'MRR', 'TCC']:
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
        
        # TCC Calculation
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

def evaluate_rgrec(model, test_data, item_to_cat, num_cats, rgrec_weight, desc="Eval"):
    model.eval()
    loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=opt.batch_size, shuffle=False)
    k_list = [5, 10]
    MIN_HISTORY_LEN = 4
    
    # 1. Initialize RGRec Manager
    if hasattr(model, 'module'):
        item_emb_weight = model.module.item_embedding.weight.detach().cpu().numpy()
    else:
        item_emb_weight = model.item_embedding.weight.detach().cpu().numpy()
        
    rgrec_agent = RGRecManager(item_emb_weight, item_to_cat, num_cats)
    
    records_base = []
    records_rgrec = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc=desc):
            inputs, mask, targets, cat_inputs, rate_inputs, user_ids = data
            inputs = trans_to_cuda(inputs).long()
            
            # Get Base SASRec Scores
            hidden = model(inputs, attention_mask=None) 
            all_logits = model.compute_scores(hidden) 
            all_logits = trans_to_cpu(all_logits).detach().numpy()
            
            inputs_np = inputs.cpu().numpy()
            cat_inputs_np = cat_inputs.numpy()
            batch_size = inputs.shape[0]
            
            for b in range(batch_size):
                u_items = inputs_np[b]
                u_cats = cat_inputs_np[b]
                
                # Extract valid history
                valid_idx = np.where(u_items != 0)[0]
                if len(valid_idx) <= MIN_HISTORY_LEN: continue
                
                # Last step prediction
                t = valid_idx[-1]
                target_item = targets[b].item()

                window_len = 20
                window_start = max(0, len(valid_idx) - window_len)
                
                # History for HD calculation
                recent_cats = u_cats[valid_idx[window_start:]]
                valid_recent_cats = [c for c in recent_cats if c < num_cats]
                
                hd_val = 0.0
                local_dominant_cat = -1
                if valid_recent_cats:
                    counts = np.bincount(valid_recent_cats)
                    local_dominant_cat = np.argmax(counts)
                    hd_val = counts[local_dominant_cat] / len(valid_recent_cats)

                history_cats_for_nudge = recent_cats
                
                # Candidate Generation (SASRec Top 200)
                step_scores = all_logits[b, t, :].copy()
                step_scores[0] = -np.inf 
                
                rerank_k = 200
                # Get indices of top 200 scores
                cand_ind = np.argpartition(step_scores, -rerank_k)[-rerank_k:]
                cand_scores = step_scores[cand_ind]
                
                # Sort candidates by SASRec score
                sorted_idx = np.argsort(-cand_scores)
                final_cands = cand_ind[sorted_idx].tolist()
                
                # --- Baseline (Pure SASRec) ---
                metrics_base = calculate_metrics_for_list(
                    final_cands, target_item, k_list, item_to_cat, local_dominant_cat
                )
                records_base.append({'risk_val': float(hd_val), 'metrics': metrics_base})
                
                # --- RGRec Method ---
                # Apply Nudging Strategy
                final_list_rgrec = rgrec_agent.rerank_feed(
                    user_history_cats=history_cats_for_nudge,
                    original_topk_items=final_cands,
                    original_scores=cand_scores, 
                    nudge_weight=rgrec_weight
                )
                
                metrics_rgrec = calculate_metrics_for_list(
                    final_list_rgrec, target_item, k_list, item_to_cat, local_dominant_cat
                )
                records_rgrec.append({'risk_val': float(hd_val), 'metrics': metrics_rgrec})

    # --- Reporting ---
    print_report(records_base, k_list, "BASE MODEL (SASRec)")
    results_rgrec = print_report(records_rgrec, k_list, f"RGRec (w={rgrec_weight})")
    
    return results_rgrec.get("Global_Recall@10", 0.0)

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

        hit = evaluate_rgrec(
            model, test_data, item_to_cat, num_cats, rgrec_weight=0.0, desc="Valid"
        )
        
        if hit > best_hit:
            best_hit = hit
            torch.save(model.state_dict(), best_model_path)
            bad_counter = 0
            print(f"New best model saved! Recall@10: {best_hit:.4f}")
        else:
            bad_counter += 1
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

    print("\n=== RGRec Evaluation Phase ===")
    print(f"RGRec Nudge Weight (w): {opt.rgrec_weight}")
    
    evaluate_rgrec(
        model, 
        test_data, 
        item_to_cat=item_to_cat, 
        num_cats=num_cats, 
        rgrec_weight=opt.rgrec_weight,
        desc="Testing"
    )

if __name__ == '__main__':
    main()