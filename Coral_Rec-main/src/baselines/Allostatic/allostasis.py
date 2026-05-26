import numpy as np
import torch

class AllostaticRegulator:
    """
    Implementation of the Allostatic Regulator.
    Calculates penalties based on category accumulation (Fatigue).
    """
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
        """
        Calculate penalty vector for a single user at a specific time step t.
        Input: 
            user_history_items: numpy array of item IDs up to t
            user_history_cats: numpy array of category IDs up to t
            current_t: current time step index
        Returns:
            penalty: Tensor [Num_Items]
        """
        # We look at the history window up to current_t
        # History items: seq[0...t]
        # Their distance from now (t+1) is: 
        # item at t has delta=1, item at t-1 has delta=2...
        
        valid_len = len(user_history_cats)
        if valid_len == 0:
            return torch.zeros(len(self.item_cat_tensor), device=self.device)

        # Time deltas: If current is t+1. 
        # The item at index i (where i <= t) has age = (t + 1) - i? 
        # Paper implies exponential decay based on recency.
        # Let's align with the previous logic: Recency vector.
        
        indices = np.arange(valid_len)
        # item at index 'current_t' is the most recent (age 0 or 1). 
        # Let's assume item at index i has age = (valid_len - 1 - i).
        ages = (valid_len - 1) - indices 
        
        # Calculate weight: I_b * exp(-lambda * age)
        weights = self.I_b * np.exp(-self.lambda_b * ages)
        
        # Aggregate weights by category
        cat_weights = np.zeros(self.num_cats)
        for cat, w in zip(user_history_cats, weights):
            if cat < self.num_cats:
                cat_weights[cat] += w
                
        # Expand to items
        cat_weights_tensor = torch.tensor(cat_weights, dtype=torch.float32, device=self.device)
        
        # Map category penalties to items
        # item_cat_tensor: [Num_Items] -> Category ID
        # We gather the penalty for each item based on its category
        item_penalties = cat_weights_tensor[self.item_cat_tensor]
        
        return item_penalties

    def get_dominant_cat_and_hd(self, recent_cats, window_size=20):
        """
        Calculates History Dominance (HD) and the Dominant Category ID.
        """
        if len(recent_cats) == 0:
            return -1, 0.0
        
        # Take the last 'window_size' items
        window = recent_cats[-window_size:]
        valid_window = [c for c in window if c < self.num_cats]
        
        if not valid_window:
            return -1, 0.0
            
        counts = np.bincount(valid_window)
        dom_cat = np.argmax(counts)
        hd_val = counts[dom_cat] / len(valid_window)
        
        return dom_cat, hd_val