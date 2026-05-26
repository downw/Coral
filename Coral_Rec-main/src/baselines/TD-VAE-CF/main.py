import time
import argparse
import pickle
import random
import os
import numpy as np
import torch
from tqdm import tqdm
from model import TD_VAE_CF, trans_to_cuda, trans_to_cpu
from utils import Data, get_num_node_unique
from log import setup_logger 

def init_seed(seed=None):
    if seed is None: seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sparse_batch(batch_inputs, num_node):
    batch_size = len(batch_inputs)
    sparse_matrix = torch.zeros((batch_size, num_node), dtype=torch.float32)
    
    for i, seq in enumerate(batch_inputs):
        valid_items = [x for x in seq if x > 0]
        if valid_items:
            sparse_matrix[i, valid_items] = 1.0
            
    return trans_to_cuda(sparse_matrix)

def evaluate(model, test_data, train_data_raw, k_list=[10, 20], intervention_params=None):
    model.eval()
    hit_all = [[] for _ in k_list]
    mrr_all = [[] for _ in k_list]
    test_users = list(test_data.inputs.keys())
    batch_size = 100
    n_batches = (len(test_users) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for b in range(n_batches):
            batch_users = test_users[b*batch_size : (b+1)*batch_size]
            input_seqs = []
            targets = []
            
            for u in batch_users:
                if u in train_data_raw:
                    input_seqs.append(train_data_raw[u])
                else:
                    input_seqs.append([])
                targets.append(test_data.inputs[u]) 
            
            input_tensor = sparse_batch(input_seqs, model.num_node)
            logits, _, _ = model(input_tensor, intervention_params)
            logits[input_tensor > 0] = -np.inf
            _, indices = torch.topk(logits, k=max(k_list))
            indices = trans_to_cpu(indices).numpy()
            
            for i, u_target in enumerate(targets):
                pred_items = indices[i]
                
                for k_idx, k in enumerate(k_list):
                    pred_k = pred_items[:k]
                    hits = 0
                    mrr = 0
                    num_relevant = 0
                    for rank, item in enumerate(pred_k):
                        if item in u_target:
                            num_relevant += 1
                            if mrr == 0:
                                mrr = 1.0 / (rank + 1)
                    if len(u_target) > 0:
                        hit_all[k_idx].append(num_relevant / len(u_target)) # Recall
                    else:
                        hit_all[k_idx].append(0)
                        
                    mrr_all[k_idx].append(mrr)

    hit_avg = [np.mean(h) * 100 for h in hit_all]
    mrr_avg = [np.mean(m) * 100 for m in mrr_all]
    
    return hit_avg, mrr_avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ML1m')
    parser.add_argument('--hiddenSize', type=int, default=200) #
    parser.add_argument('--epoch', type=int, default=50) 
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--total_anneal_steps', type=int, default=200000)
    parser.add_argument('--anneal_cap', type=float, default=0.2)
    parser.add_argument('--intervention_lambda', type=float, default=0.5, help='strength')
    
    opt = parser.parse_args()
    init_seed(2024)

    if not os.path.exists('../../../../datasets/' + opt.dataset):
        # Fallback for demo
        print("Dataset path not found, please check path.")
        return 

    train_data_raw = pickle.load(open(os.path.join('../../../../datasets/', opt.dataset, 'processed_data/train.txt'), "rb"))
    test_data_raw = pickle.load(open(os.path.join('../../../../datasets/', opt.dataset, 'processed_data/test.txt'), "rb"))
    
    num_node = max(get_num_node_unique(train_data_raw), get_num_node_unique(test_data_raw)) + 1
    train_data = Data(train_data_raw, num_node, max_len=200, mask_prob=0.0, mode='train')
    test_data = Data(test_data_raw, num_node, max_len=200, mask_prob=0.0, mode='test')
    model = trans_to_cuda(TD_VAE_CF(opt, num_node))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    print("Start Training TD-VAE-CF...")
    update_count = 0
    intervention_direction = torch.zeros(opt.hiddenSize // 2)
    intervention_direction[0] = 1.0 
    intervention_params = {
        'direction': intervention_direction,
        'lambda': opt.intervention_lambda
    }

    best_recall = 0.0
    
    for epoch in range(opt.epoch):
        model.train()
        total_loss = 0.0
        users = list(train_data.inputs.keys())
        random.shuffle(users)
        
        n_batches = (len(users) + opt.batch_size - 1) // opt.batch_size
        
        for b in tqdm(range(n_batches), desc=f"Epoch {epoch+1}"):
            batch_users = users[b*opt.batch_size : (b+1)*opt.batch_size]
            batch_seqs = [train_data.inputs[u] for u in batch_users]
            x = sparse_batch(batch_seqs, num_node) # (Batch, Items)
            
            # Annealing Logic
            if opt.total_anneal_steps > 0:
                anneal = min(opt.anneal_cap, 1. * update_count / opt.total_anneal_steps)
            else:
                anneal = opt.anneal_cap
            update_count += 1
            
            optimizer.zero_grad()
            logits, mu, logvar = model(x)
            
            loss = model.loss_function(logits, x, mu, logvar, anneal)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {total_loss / n_batches:.4f}")
        
        if (epoch + 1) % 5 == 0:
            print("Evaluating...")
            rec, mrr = evaluate(model, test_data, train_data_raw, k_list=[10, 20])
            print(f"Original -> Recall@20: {rec[1]:.4f}, MRR@20: {mrr[1]:.4f}")
            rec_int, mrr_int = evaluate(model, test_data, train_data_raw, k_list=[10, 20], 
                                       intervention_params=intervention_params)
            print(f"Intervened -> Recall@20: {rec_int[1]:.4f}, MRR@20: {mrr_int[1]:.4f}")
            if rec[1] > best_recall:
                best_recall = rec[1]
                # torch.save(model.state_dict(), f'td_vae_best_{opt.dataset}.pth')

if __name__ == '__main__':
    main()