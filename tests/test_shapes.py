import pytest
import torch 
from src.models.attention import MultiheadAttention


#this moved in conftest.py
# # fixtures 
# @pytest.fixture
# def batch_size():
#     return 2

# @pytest.fixture
# def seq_len():
#     return 16

# @pytest.fixture
# def embed_dim():
#     return 128

# @pytest.fixture
# def num_heads():
#     return 8

# @pytest.fixture
# def dummy_input(batch_size, seq_len, embed_dim):
#     return torch.randn(batch_size, seq_len, embed_dim)


# MultiheadAttention Tests
def test_multihead_attention_output_shape(dummy_input, embed_dim, num_heads):
    attention = MultiheadAttention(embed_dim, num_heads)
    output, attn_weights = attention(dummy_input, return_attention=True)
    assert output.shape == dummy_input.shape, f"Expected output shape {dummy_input.shape}, but got {output.shape}"

    expected_attn_shape = (dummy_input.shape[0], num_heads, dummy_input.shape[1], dummy_input.shape[1])
    assert attn_weights.shape == expected_attn_shape, f"Expected attention weights shape {expected_attn_shape}, but got {attn_weights.shape}"

def test_multihead_attention_with_mask(dummy_input, embed_dim, num_heads):
    attention = MultiheadAttention(embed_dim, num_heads)
    mask = torch.ones(dummy_input.shape[0], dummy_input.shape[1])  # (batch_size, seq_len)
    # mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
    output, attn_weights = attention(dummy_input, mask=mask, return_attention=True)
    assert output.shape == dummy_input.shape, f"Expected output shape {dummy_input.shape}, but got {output.shape}"

    expected_attn_shape = (dummy_input.shape[0], num_heads, dummy_input.shape[1], dummy_input.shape[1])
    assert attn_weights.shape == expected_attn_shape, f"Expected attention weights shape {expected_attn_shape}, but got {attn_weights.shape}"
