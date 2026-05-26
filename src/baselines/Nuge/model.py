import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import sys

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


class SASRec(nn.Module):
    def __init__(self, opt, num_node):
        super(SASRec, self).__init__()
        self.opt = opt
        self.batch_size = opt.batch_size
        self.num_node = num_node 
        self.dim = opt.hiddenSize
        self.n_layers = getattr(opt, 'n_layers', 2)
        self.n_heads = getattr(opt, 'n_heads', 2)
        self.dropout = getattr(opt, 'dropout_local', 0.2)
        self.max_len = getattr(opt, 'max_len', 200)
        
        self.item_embedding = nn.Embedding(self.num_node + 1, self.dim, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_len, self.dim)
        self.emb_dropout = nn.Dropout(self.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.n_heads,
            dim_feedforward=self.dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.layer_norm = nn.LayerNorm(self.dim)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=opt.lr_dc_step, 
            gamma=opt.lr_dc
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, inputs, attention_mask):
        seq_length = inputs.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_ids = position_ids.unsqueeze(0).expand_as(inputs)
        
        item_emb = self.item_embedding(inputs)
        pos_emb = self.position_embedding(position_ids)
        
        x = item_emb + pos_emb
        x = self.emb_dropout(x)
        
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, device=inputs.device), diagonal=1).bool()
        padding_mask = (inputs == 0)
        
        output = self.transformer_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        return output

    def compute_scores(self, hidden):
        test_item_emb = self.item_embedding.weight 
        scores = torch.matmul(hidden, test_item_emb.transpose(0, 1)) 
        scores[:, 0] = -np.inf 
        return scores

def forward(model, data, mode='train'):
    if mode == 'train':
        inputs, mask, targets = data[0], data[1], data[2]
        inputs = trans_to_cuda(inputs).long()
        targets = trans_to_cuda(targets).long()
        
        hidden = model(inputs, attention_mask=None) 
        scores = model.compute_scores(hidden)
        
        scores = scores.view(-1, scores.size(-1))
        targets = targets.view(-1)
        return targets, scores
    else:
        inputs = data[0]
        inputs = trans_to_cuda(inputs).long()
        hidden = model(inputs, attention_mask=None) 
        
        seq_lens = (inputs != 0).sum(dim=1) - 1 
        batch_size = hidden.size(0)
        last_hidden = hidden[torch.arange(batch_size, device=hidden.device), seq_lens] 
        
        scores = model.compute_scores(last_hidden) 
        return scores