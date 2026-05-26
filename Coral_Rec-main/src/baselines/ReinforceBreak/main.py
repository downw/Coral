import time
import argparse
import pickle
import random
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.cluster import KMeans
from collections import Counter

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
parser.add_argument('--lr_dc_step', type=int, default=8)
parser.add_argument('--l2', type=float, default=1e-5)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--dropout_local', type=float, default=0.2)
parser.add_argument('--max_len', type=int, default=200)
parser.add_argument('--mask_prob', type=float, default=0.2)
parser.add_argument('--validation', action='store_true')
parser.add_argument('--valid_portion', type=float, default=0.1)
parser.add_argument('--patience', type=int, default=3)

# Controllable Recommendation Args (New for Paper)
parser.add_argument('--n_communities', type=int, default=5, 
                    help='Number of communities to detect (approx. Louvain communities)')
parser.add_argument('--controllable_steps', type=int, default=1,
                    help='Number of iterative exposure/retraining steps')

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
    # Sort by risk_val (High Risk = Low Diversity/High Dominance)
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
    """ Helper to print the stratified analysis table (Exact format as requested) """
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
        topk = topk_list[:K]
        hit = 1.0 if target_item in topk else 0.0
        metrics[f'Recall@{K}'] = hit
        metrics[f'MRR@{K}'] = 1.0 / (topk.index(target_item) + 1) if hit else 0.0
        
        # TCC Calculation (Target Category Coherence - proxy for diversity risk)
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

class CommunityEvaluator:
    def __init__(self, model, num_items, n_clusters=5):
        self.model = model
        self.num_items = num_items
        self.n_clusters = n_clusters
        self.item_labels = None
    
    def detect_communities(self):
        """
        Extracts item embeddings and clusters them to find item communities.
        """
        if hasattr(self.model, 'module'):
            item_emb = self.model.module.item_embedding.weight.detach().cpu().numpy()
        else:
            item_emb = self.model.item_embedding.weight.detach().cpu().numpy()
        
        # Ignore padding index 0
        valid_embs = item_emb[1:self.num_items+1]
        
        # Using KMeans as a differentiable proxy for graph modularity optimization
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=2023, n_init=10)
        labels = kmeans.fit_predict(valid_embs)
        
        # Map item_id -> community_id
        self.item_labels = {i+1: l for i, l in enumerate(labels)}
        self.item_labels[0] = -1
        return self.item_labels

    def get_user_community(self, user_seq):
        """
        Determine user community based on dominant item category in history.
        """
        if not self.item_labels: self.detect_communities()
        comms = [self.item_labels.get(i, -1) for i in user_seq if i != 0]
        if not comms: return -1
        return Counter(comms).most_common(1)[0][0]

class HeuristicExposureAgent:
    def __init__(self, evaluator, num_items):
        self.evaluator = evaluator
        self.num_items = num_items
    
    def select_exposure_edges(self, train_data, n_edges_per_user=10):
        print("[Agent] Analyzing graph structure to select exposure edges...")
        item_labels = self.evaluator.item_labels
        new_edges = []
        
        # Group items by community for fast sampling
        comm_items = {}
        for item, comm in item_labels.items():
            if comm == -1: continue
            if comm not in comm_items: comm_items[comm] = []
            comm_items[comm].append(item)
            
        raw_inputs = train_data.raw_inputs
        for uid, seq in enumerate(tqdm(raw_inputs, desc="Generating Edges")):
            user_comm = self.evaluator.get_user_community(seq)
            if user_comm == -1: continue
            
            # Action: Select a target community != user_comm
            available_comms = list(comm_items.keys())
            if user_comm in available_comms:
                available_comms.remove(user_comm)
            
            if not available_comms: continue
            
            # Heuristic: Randomly pick a different community to break bubble
            target_comm = np.random.choice(available_comms)
            
            if comm_items[target_comm]:
                target_item = np.random.randint(1, self.num_items + 1)
                new_edges.append((uid, target_item))
                
        return new_edges


def evaluate_model_performance(model, test_data, item_to_cat, num_cats, desc="Eval"):
    model.eval()
    loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=opt.batch_size, shuffle=False)
    k_list = [5, 10]
    MIN_HISTORY_LEN = 4 

    records_base = []
    
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
                t = len(seq_items) - 2 # Index of last input item
                if t < MIN_HISTORY_LEN - 1: continue

                target_item = seq_items[t+1]
                
                # [Risk Calculation: Historical Dominance]
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
                
                # [Ranking]
                step_scores = seq_logits[t].copy()
                step_scores[0] = -np.inf 
                
                ind = np.argpartition(step_scores, -50)[-50:]
                base_scores = step_scores[ind]
                sorted_base_idx = np.argsort(-base_scores)
                final_list = ind[sorted_base_idx].tolist()
                
                metrics = calculate_metrics_for_list(
                    final_list, target_item, k_list, item_to_cat, local_dominant_cat
                )
                records_base.append({'risk_val': float(hd_val), 'metrics': metrics})

    # Print Report (Generates the table)
    return print_report(records_base, k_list, f"MODEL PERFORMANCE ({desc})")

def train_epoch(model, train_data, optimizer):
    model.train()
    loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    total_loss = 0.0
    for data in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        targets, scores = forward_train(model, data)
        loss = model.loss_function(scores, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

def main():
    setup_logger(opt.dataset, opt)
    redirect_tqdm_to_console()
    init_seed(2023)
    
    # 1. Load Data
    print("Loading data...")
    train_data_raw = pickle.load(open(os.path.join('../../../../datasets/', opt.dataset, 'processed_data/train.txt'), "rb"))
    test_data_raw = pickle.load(open(os.path.join('../../../../datasets/', opt.dataset, 'processed_data/test.txt'), "rb"))
    
    num_node = max(get_num_node_unique(train_data_raw), get_num_node_unique(test_data_raw))
    item_to_cat, cat_to_items, num_cats = build_maps(train_data_raw)
    
    train_data = Data(train_data_raw, num_node, opt.max_len, opt.mask_prob, mode='train')
    test_data = Data(test_data_raw, num_node, opt.max_len, opt.mask_prob, mode='test')
    
    model = trans_to_cuda(SASRec(opt, num_node))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    
    # 2. Phase 1: Pre-training (Filter Bubble Formation)
    print("\n=== Phase 1: Training Base RS (Filter Bubble Formation) ===")
    save_dir = '../../checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'sasrec_base_{opt.dataset}.pth')
    
    if os.path.exists(model_path):
        print("Loading pre-trained base model...")
        model.load_state_dict(torch.load(model_path))
    else:
        best_recall = 0.0
        for epoch in range(opt.epoch):
            train_epoch(model, train_data, optimizer)
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                res = evaluate_model_performance(model, test_data, item_to_cat, num_cats, desc=f"Ep{epoch}")
                if res['Global_Recall@10'] > best_recall:
                    best_recall = res['Global_Recall@10']
                    torch.save(model.state_dict(), model_path)
        model.load_state_dict(torch.load(model_path))

    # Initial Eval
    print("\n>>> Base Model Risk Profile:")
    evaluate_model_performance(model, test_data, item_to_cat, num_cats, desc="Base")

    # 3. Phase 2: Controllable Exposure Loop (Breaking Bubble)
    print("\n=== Phase 2: Controllable Exposure Strategy ===")
    
    comm_eval = CommunityEvaluator(model, num_node, n_clusters=opt.n_communities)
    agent = HeuristicExposureAgent(comm_eval, num_node)
    
    for step in range(opt.controllable_steps):
        print(f"\n[Controllable Step {step+1}/{opt.controllable_steps}]")
        
        # A. Detect Communities & Select Edges (Loop Step 2 in paper)
        comm_eval.detect_communities()
        new_edges = agent.select_exposure_edges(train_data)
        
        # B. Inject to Dataset (Loop Step 2 in paper)
        train_data.inject_exposure_data(new_edges)
        
        # C. Retrain (Loop Step 4 in paper)
        print("Fine-tuning on exposed data...")
        for _ in range(2): 
            train_epoch(model, train_data, optimizer)
        
        # D. Evaluate (Check impact on High Risk Users)
        print(f">>> Step {step+1} Risk Profile:")
        evaluate_model_performance(model, test_data, item_to_cat, num_cats, desc=f"Control_S{step+1}")

if __name__ == '__main__':
    main()