import numpy as np
import torch
import random
from torch.utils.data import Dataset

def get_num_node_unique(data_list):
    """
    Count unique items in the dataset to determine embedding size.
    """
    sequences, _, _, targets, _, _ = data_list
    item_set = set()
    for seq in sequences:
        item_set.update(seq)
    item_set.update(targets)
    if 0 in item_set:
        item_set.remove(0)
    return len(item_set)

class Data(Dataset):
    def __init__(self, raw_data, num_items, max_len, mask_prob, mode='train'):
        self.raw_inputs = raw_data[0]
        self.raw_cats = raw_data[2]    
        self.raw_targets = raw_data[3]
        self.raw_ratings = raw_data[5] 
        
        self.num_items = num_items
        self.max_len = max_len
        self.mode = mode
        
        processed = self._prepare_data()
        self.inputs = np.asarray(processed[0])
        self.mask = np.asarray(processed[1])
        self.targets = np.asarray(processed[2])
        self.cat_inputs = np.asarray(processed[3])
        self.rate_inputs = np.asarray(processed[4])
        self.user_ids = np.asarray(processed[5]) 
        
        self.length = len(self.inputs)

    def _prepare_data(self):
        new_inputs = []
        new_masks = []
        new_targets = []
        new_cats = []
        new_rates = []
        new_users = []
        
        for i in range(len(self.raw_inputs)):
            seq = self.raw_inputs[i]
            cat_seq = self.raw_cats[i]
            rate_seq = self.raw_ratings[i]
            
            seq_clean = []
            cat_clean = []
            rate_clean = []
            
            limit = min(len(seq), len(cat_seq), len(rate_seq))
            for idx in range(limit):
                item = seq[idx]
                if item != 0:
                    seq_clean.append(item)
                    cat_clean.append(cat_seq[idx])
                    rate_clean.append(rate_seq[idx])
            
            if len(seq_clean) < 2:
                continue
                
            if self.mode == 'train':
                if len(seq_clean) > self.max_len + 1:
                    seq_clean = seq_clean[-(self.max_len + 1):]
                    cat_clean = cat_clean[-(self.max_len + 1):]
                    rate_clean = rate_clean[-(self.max_len + 1):]
                
                input_seq = seq_clean[:-1]
                target_seq = seq_clean[1:]
                input_cat = cat_clean[:-1]
                input_rate = rate_clean[:-1]
                
                pad_len = self.max_len - len(input_seq)
                if pad_len > 0:
                    input_seq = input_seq + [0] * pad_len
                    target_seq = target_seq + [0] * pad_len
                    input_cat = input_cat + [0] * pad_len
                    input_rate = input_rate + [0] * pad_len
                else:
                    input_seq = input_seq[-self.max_len:]
                    target_seq = target_seq[-self.max_len:]
                    input_cat = input_cat[-self.max_len:]
                    input_rate = input_rate[-self.max_len:]
                
                new_inputs.append(input_seq)
                new_targets.append(target_seq)
                new_masks.append([1 if x != 0 else 0 for x in input_seq])
                new_cats.append(input_cat)
                new_rates.append(input_rate)
                new_users.append(i)
                
            else:
                target_item = self.raw_targets[i]
                input_seq = seq_clean
                input_cat = cat_clean
                input_rate = rate_clean
                
                if len(input_seq) > self.max_len:
                    input_seq = input_seq[-self.max_len:]
                    input_cat = input_cat[-self.max_len:]
                    input_rate = input_rate[-self.max_len:]
                
                pad_len = self.max_len - len(input_seq)
                mask_seq = [1] * len(input_seq) + [0] * pad_len
                
                input_seq = input_seq + [0] * pad_len
                input_cat = input_cat + [0] * pad_len
                input_rate = input_rate + [0] * pad_len
                
                new_inputs.append(input_seq)
                new_targets.append(target_item)
                new_masks.append(mask_seq)
                new_cats.append(input_cat)
                new_rates.append(input_rate)
                new_users.append(i)
                
        return [new_inputs, new_masks, new_targets, new_cats, new_rates, new_users]

    def __getitem__(self, index):
        return [
            torch.tensor(self.inputs[index], dtype=torch.long), 
            torch.tensor(self.mask[index], dtype=torch.long), 
            torch.tensor(self.targets[index], dtype=torch.long),
            torch.tensor(self.cat_inputs[index], dtype=torch.long),
            torch.tensor(self.rate_inputs[index], dtype=torch.float),
            torch.tensor(self.user_ids[index], dtype=torch.long)
        ]

    def __len__(self):
        return self.length

class RGRecManager:
    def __init__(self, item_emb_weight, item_to_cat, num_cats):
        self.item_emb = item_emb_weight
        self.item_to_cat = item_to_cat
        self.num_cats = num_cats
        self.cat_emb = self._compute_topic_embeddings()
        self.topic_sim_matrix = self._compute_topic_similarity()

    def _compute_topic_embeddings(self):
        """ Eq (3) Concept: Aggregating item embeddings to get topic embedding. """
        cat_vectors = {}
        counts = {}
        
        # Aggregate
        for item_id, cat_id in self.item_to_cat.items():
            if item_id >= len(self.item_emb): continue
            vec = self.item_emb[item_id]
            if cat_id not in cat_vectors:
                cat_vectors[cat_id] = np.zeros_like(vec)
                counts[cat_id] = 0
            cat_vectors[cat_id] += vec
            counts[cat_id] += 1
            
        # Average
        final_cat_emb = np.zeros((self.num_cats, self.item_emb.shape[1]))
        for c in range(self.num_cats):
            if c in cat_vectors and counts[c] > 0:
                final_cat_emb[c] = cat_vectors[c] / counts[c]
            else:
                # Random init if category has no items in map (rare)
                final_cat_emb[c] = np.random.normal(0, 0.1, self.item_emb.shape[1])
        
        # Normalize
        norm = np.linalg.norm(final_cat_emb, axis=1, keepdims=True)
        return final_cat_emb / (norm + 1e-8)

    def _compute_topic_similarity(self):
        """ Eq (3): Cosine similarity between topics. """
        return np.dot(self.cat_emb, self.cat_emb.T) # [num_cats, num_cats]

    def _get_user_belief(self, history_cats):
        if len(history_cats) == 0:
            return np.zeros(self.num_cats)
            
        counts = np.bincount(history_cats, minlength=self.num_cats)
        total = np.sum(counts)
        r_ix = counts / (total + 1e-9) 
        return r_ix 

    def get_nudge_path(self, belief_dist, sp_node=None, lp_node=None):
        if sp_node is None:
            sp_node = np.argmax(belief_dist)
        
        if lp_node is None:
            # Find a topic with low (but non-zero if possible, or zero) belief
            lp_candidates = np.argsort(belief_dist) # ascending
            lp_node = lp_candidates[0] # The least interacted topic
            
        path = [sp_node]
        curr = sp_node
        max_steps = 3 
        
        visited = {sp_node}
        
        for _ in range(max_steps):
            if curr == lp_node:
                break
                
            sims = self.topic_sim_matrix[curr]
            
            # Heuristic: Pick neighbor that is most similar to Current 
            # but also has some similarity to LP (guiding heuristic).
            scores = sims + 0.5 * self.topic_sim_matrix[lp_node]
            candidates = np.argsort(-scores)
            
            found = False
            for cand in candidates:
                if cand not in visited:
                    path.append(cand)
                    visited.add(cand)
                    curr = cand
                    found = True
                    break
            if not found:
                break
                
        if lp_node not in visited:
            path.append(lp_node)
            
        return path

    def rerank_feed(self, user_history_cats, original_topk_items, original_scores, nudge_weight=0.6):
        """
        RecomGen Module Logic:
        Mix Feed_original (SASRec) with G_I (Items from Nudge Path).
        """
        belief = self._get_user_belief(user_history_cats)
        sp_node = np.argmax(belief)
        lp_node = np.argmin(belief)
        path_topics = self.get_nudge_path(belief, sp_node, lp_node)
        
        feed_original = []
        feed_nudge = []
        
        path_set = set(path_topics)
        if sp_node in path_set and len(path_set) > 1:
            path_set.remove(sp_node)
            
        for idx, item in enumerate(original_topk_items):
            cat = self.item_to_cat.get(item, -1)
            if cat in path_set:
                feed_nudge.append(item)
            else:
                feed_original.append(item)
                
        # Mixing
        K = 20 
        n_nudge = int(K * nudge_weight)
        n_orig = K - n_nudge
        
        final_list = []
        
        added_nudge = feed_nudge[:n_nudge]
        added_orig = feed_original[:n_orig]
        
        final_list.extend(added_nudge)
        final_list.extend(added_orig)
        
        remain_nudge = feed_nudge[n_nudge:]
        remain_orig = feed_original[n_orig:]
        
        if len(final_list) < K:
            needed = K - len(final_list)
            pool = remain_orig + remain_nudge
            final_list.extend(pool[:needed])
            
        return final_list[:K]