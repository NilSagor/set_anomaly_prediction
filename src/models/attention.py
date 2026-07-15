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