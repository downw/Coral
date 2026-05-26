import numpy as np
import torch

class AllostaticRegulator:
    def __init__(self, item_to_cat, num_cats, I_b, lambda_b, device='cuda'):
        self.item_to_cat = item_to_cat
        self.num_cats = num_cats
        self.I_b = I_b
        self.lambda_b = lambda_b
        self.device = device
        
        # Fast lookup tensor
        max_item = max(item_to_cat.keys()) if item_to_cat else 0
        self.item_cat_tensor = torch.zeros(max_item + 1, dtype=torch.long, device=device)
        for item, cat in item_to_cat.items():
            self.item_cat_tensor[item] = cat

    def get_step_penalty(self, user_history_items, user_history_cats, current_t):
        
        valid_len = len(user_history_cats)
        if valid_len == 0:
            return torch.zeros(len(self.item_cat_tensor), device=self.device)

        indices = np.arange(valid_len)
        ages = (valid_len - 1) - indices 
        weights = self.I_b * np.exp(-self.lambda_b * ages)
        cat_weights = np.zeros(self.num_cats)
        for cat, w in zip(user_history_cats, weights):
            if cat < self.num_cats:
                cat_weights[cat] += w
                
        # Expand to items
        cat_weights_tensor = torch.tensor(cat_weights, dtype=torch.float32, device=self.device)
        item_penalties = cat_weights_tensor[self.item_cat_tensor]
        
        return item_penalties

    def get_dominant_cat_and_hd(self, recent_cats, window_size=20):
        if len(recent_cats) == 0:
            return -1, 0.0
        window = recent_cats[-window_size:]
        valid_window = [c for c in window if c < self.num_cats]
        
        if not valid_window:
            return -1, 0.0
            
        counts = np.bincount(valid_window)
        dom_cat = np.argmax(counts)
        hd_val = counts[dom_cat] / len(valid_window)
        
        return dom_cat, hd_val