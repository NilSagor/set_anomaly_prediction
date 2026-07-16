import pytest 
import torch 



import pytest
import torch


@pytest.fixture(scope="session")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 2


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def embed_dim():
    return 128


@pytest.fixture
def num_heads():
    return 8


@pytest.fixture
def head_dim(embed_dim, num_heads):
    assert embed_dim % num_heads == 0
    return embed_dim // num_heads


@pytest.fixture
def dummy_input(batch_size, seq_len, embed_dim, device):
    """Reproducible random input."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, embed_dim, device=device)


@pytest.fixture
def attention(embed_dim, num_heads, device):
    from src.models.attention import MultiheadAttention
    attn = MultiheadAttention(embed_dim, num_heads).to(device)
    return attn


@pytest.fixture
def full_mask(batch_size, seq_len, device):
    """A mask that allows all positions."""
    return torch.ones(batch_size, seq_len, device=device)


@pytest.fixture
def partial_mask(batch_size, seq_len, device):
    """Masks out the last token for every sequence."""
    mask = torch.ones(batch_size, seq_len, device=device)
    mask[:, -1] = 0
    return mask