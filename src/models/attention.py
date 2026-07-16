import torch 
import torch.nn as nn
import math

import torch.nn.functional as F
import lightning as L

# def scaled_dot_product(q,k,v, mask=None):
#     d_k = q.size()[-1]
#     attn_logits = torch.matmul(q,k.transpose(-2,-1))
#     attn_logits  = attn_logits/math.sqrt(d_k)
#     if mask is not None:
#         attn_logits = attn_logits.masked_fill(mask==0, -9e15)
#     attention = F.softmax(attn_logits, dim=-1)
#     values = torch.matmul(attention, v)
#     return values, attention


def scaled_dot_product(q, k, v, mask=None, dropout_p=0.0):
    # Use native PyTorch SDPA for efficiency (Flash Attention)
    attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
    
    # Compute weights separately for visualisation (no gradients to save memory)
    with torch.no_grad():
        d_k = q.size(-1)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -9e15)
        attn_weights = F.softmax(logits, dim=-1)
    
    return attn_output, attn_weights


def expand_mask(mask):
    # Output shape supports (batch_size, number of heads, seq length, seq length)
    # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
    assert mask.ndim >= 2, "Mask must be of shape (batch_size, seq_len)"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask



class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3*embed_dim, bias = bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias = bias)

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.constant_(self.qkv_proj.bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)   
    
    def forward(self, x, mask=None, return_attention=False):
        # x: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()
        
        if mask is not None:
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)

        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3*embed_dim)
        
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3*self.head_dim)  # (batch_size, seq_len, num_heads, 3*head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, 3*head_dim)
        q, k, v = qkv.chunk(3, dim=-1)  

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        values = values.view(batch_size, seq_len, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        output = self.out_proj(values)  # (batch_size, seq_len, embed_dim)
        
        if return_attention:
            return output, attention
        else:
            return output
        







# seq_len, d_k = 3, 2
# L.seed_everything(42)
# q = torch.randn(seq_len, d_k)
# k = torch.randn(seq_len, d_k)
# v = torch.randn(seq_len, d_k)
# values, attention = scaled_dot_product(q, k, v)
# print("Q\n", q)
# print("K\n", k)
# print("V\n", v)
# print("Values\n", values)
# print("Attention\n", attention)