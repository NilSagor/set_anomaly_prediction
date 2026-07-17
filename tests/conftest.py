import pytest 
import torch 

from src.models.modules import PositionalEncoding, EncoderBlock, TransformerEncoder
from src.lightning_module.predictor import TransformerPredictor, SetAnomalyPredictor

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
def dim_feedforward():
    # feed-forward dimension is often 2-4 times the embedding dimension for encoder blocks
    return 256

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


@pytest.fixture
def pos_enc(embed_dim, device):
    from src.models.modules import PositionalEncoding
    return PositionalEncoding(embed_dim, max_len=5000).to(device)


@pytest.fixture
def encoder_block(embed_dim, num_heads, dim_feedforward, device):
    from src.models.modules import EncoderBlock
    return EncoderBlock(embed_dim, num_heads, dim_feedforward).to(device)


@pytest.fixture
def transformer_encoder(embed_dim, num_heads, dim_feedforward, device):
    from src.models.modules import TransformerEncoder
    return TransformerEncoder(embed_dim, num_heads, dim_feedforward, num_layers=2).to(device)

# Predictor

@pytest.fixture
def num_classes():
    return 3


@pytest.fixture
def dummy_batch(batch_size, seq_len, embed_dim, device):
    """A batch tuple: (img_sets, mask, labels) for SetAnomalyPredictor."""
    torch.manual_seed(7)
    img_sets = torch.randn(batch_size, seq_len, embed_dim, device=device)
    mask = torch.ones(batch_size, seq_len, device=device)
    # Binary labels per sequence element
    labels = torch.randint(0, 2, (batch_size, seq_len), device=device)
    return img_sets, mask, labels


@pytest.fixture
def transformer_predictor(embed_dim, num_heads, device):
    """Instantiate TransformerPredictor with default hyperparameters."""
    return TransformerPredictor(
        input_dim=embed_dim,
        model_dim=embed_dim,
        num_classes=3,
        num_heads=num_heads,
        num_layers=2,
        lr=1e-3,
        dropout=0.1,
        input_dropout=0.0,
    ).to(device)


@pytest.fixture
def set_anomaly_predictor(embed_dim, num_heads, device):
    """Instantiate SetAnomalyPredictor with default hyperparameters."""
    return SetAnomalyPredictor(
        input_dim=embed_dim,
        model_dim=embed_dim,
        num_classes=2,          # binary anomaly detection
        num_heads=num_heads,
        num_layers=2,           # SAB layers count
        lr=1e-3,
        dropout=0.1,
        input_dropout=0.0,
    ).to(device)