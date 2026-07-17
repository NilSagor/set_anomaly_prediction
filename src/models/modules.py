import torch
import torch.nn as nn
import math
from src.models.attention import MultiheadAttention



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)

        # Create a long enough P matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model: cos slice may be one element shorter
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        if x.size(1) > self.pe.size(1):
            raise IndexError(
                f"Sequence length {x.size(1)} exceeds maximum length {self.pe.size(1)}"
            )
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)




class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout = 0.0):
        super().__init__()
        self.self_attn = MultiheadAttention(input_dim, num_heads, dropout=dropout)  

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attention=False):
        # Self-attention
        attn_output, attn_weights = self.self_attn(x, mask=mask, return_attention=True)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feedforward network
        ff_output = self.linear_net(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        if return_attention:
            return x, attn_weights
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, num_layers, dropout = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(input_dim, num_heads, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        return x
    
    # def get_attention_weights(self, x, mask=None):
    #     """Return attention weights from all layers."""
    #     attn_weights_list = []
    #     for layer in self.layers:
    #         _, attn_weights = layer.self_attn(x, mask=mask, return_attention=True)
    #         attn_weights_list.append(attn_weights)
    #         x = layer(x, mask=mask)  # forward pass to get the next input
    #     return attn_weights_list

    def get_attention_maps(self, x, mask=None):
        # get_attention_weights
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask=mask, return_attention=True)
            attn_weights_list.append(attn_weights)
        return attn_weights_list