
import torch
import sys
from pathlib import Path

# Add project root to path if running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.attention import MultiheadAttention
from src.models.modules import PositionalEncoding, EncoderBlock, TransformerEncoder
from src.lightning_module.predictor import TransformerPredictor, SetAnomalyPredictor


def test_multihead_attention():
    """Instantiate MultiheadAttention and run a forward pass."""
    model = MultiheadAttention(embed_dim=128, num_heads=8)
    x = torch.randn(2, 16, 128)
    out, attn = model(x, return_attention=True)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert attn.shape == (2, 8, 16, 16), f"Expected (2,8,16,16), got {attn.shape}"
    print(" MultiheadAttention")


def test_positional_encoding():
    """Instantiate PositionalEncoding and run a forward pass."""
    model = PositionalEncoding(d_model=128, max_len=5000)
    x = torch.randn(2, 16, 128)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(" PositionalEncoding")


def test_encoder_block():
    """Instantiate EncoderBlock and run a forward pass."""
    model = EncoderBlock(input_dim=128, num_heads=8, dim_feedforward=256)
    x = torch.randn(2, 16, 128)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(" EncoderBlock")


def test_transformer_encoder():
    """Instantiate TransformerEncoder and run a forward pass."""
    model = TransformerEncoder(
        input_dim=128,
        num_heads=8,
        dim_feedforward=256,
        num_layers=2
    )
    x = torch.randn(2, 16, 128)
    out = model(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(" TransformerEncoder")


def test_transformer_predictor():
    """Instantiate TransformerPredictor and run a forward pass."""
    model = TransformerPredictor(
        input_dim=128,
        model_dim=128,
        num_classes=3,
        num_heads=8,
        num_layers=2,
        lr=1e-3,
        dropout=0.1,
        input_dropout=0.0,
    )
    x = torch.randn(2, 16, 128)
    out = model(x, add_positional_encoding=True)
    expected = (2, 3)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print(" TransformerPredictor")


def test_set_anomaly_predictor():
    """Instantiate SetAnomalyPredictor and run a forward pass."""
    model = SetAnomalyPredictor(
        input_dim=128,
        model_dim=128,
        num_classes=2,
        num_heads=8,
        num_layers=2,
        lr=1e-3,
        dropout=0.1,
        input_dropout=0.0,
    )
    x = torch.randn(2, 16, 128)
    out = model(x, add_positional_encoding=False)
    expected = (2, 16, 2)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print(" SetAnomalyPredictor")


if __name__ == "__main__":
    print("Running smoke tests...")
    test_multihead_attention()
    test_positional_encoding()
    test_encoder_block()
    test_transformer_encoder()
    test_transformer_predictor()
    test_set_anomaly_predictor()
    print("\n All smoke tests passed!")