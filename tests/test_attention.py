import pytest
import torch

from src.models.attention import MultiheadAttention


# =====================================================================
# 1. SHAPE & CONTRACT
# =====================================================================

def test_output_shape(dummy_input, attention, num_heads):
    output, attn_weights = attention(dummy_input, return_attention=True)
    assert output.shape == dummy_input.shape, \
        f"Expected output shape {dummy_input.shape}, got {output.shape}"

    b, l, _ = dummy_input.shape
    expected_attn = (b, num_heads, l, l)
    assert attn_weights.shape == expected_attn, \
        f"Expected attn shape {expected_attn}, got {attn_weights.shape}"


def test_dtype_and_device_preserved(dummy_input, attention, device):
    output, attn_weights = attention(dummy_input)
    assert output.device == device
    assert attn_weights.device == device
    assert output.dtype == dummy_input.dtype


# =====================================================================
# 2. BEHAVIORAL INVARIANTS (not just shapes)
# =====================================================================

def test_attention_weights_are_probabilities(dummy_input, attention):
    """Attention weights must be non-negative and sum to 1 across keys."""
    _, attn_weights = attention(dummy_input, return_attention=True)

    assert (attn_weights >= 0).all(), "Attention weights contain negative values"
    assert not torch.isnan(attn_weights).any(), "Attention weights contain NaN"
    assert not torch.isinf(attn_weights).any(), "Attention weights contain Inf"

    sums = attn_weights.sum(dim=-1)
    expected = torch.ones_like(sums)
    assert torch.allclose(sums, expected, atol=1e-5), \
        "Attention weights do not sum to 1 across the key dimension"


def test_output_is_finite(dummy_input, attention):
    output, _ = attention(dummy_input, return_attention=True)
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"


def test_forward_is_deterministic_in_eval(dummy_input, attention):
    """Eval mode must produce identical outputs for identical inputs."""
    attention.eval()
    with torch.no_grad():
        out1, attn1 = attention(dummy_input, return_attention=True)
        out2, attn2 = attention(dummy_input, return_attention=True)

    assert torch.allclose(out1, out2, atol=1e-6), "Eval forward is non-deterministic"
    assert torch.allclose(attn1, attn2, atol=1e-6), "Eval attention is non-deterministic"


def test_output_changes_with_input(attention, batch_size, seq_len, embed_dim, device):
    """Different inputs must produce different outputs (sanity check)."""
    torch.manual_seed(1)
    x1 = torch.randn(batch_size, seq_len, embed_dim, device=device)
    torch.manual_seed(2)
    x2 = torch.randn(batch_size, seq_len, embed_dim, device=device)

    attention.eval()
    with torch.no_grad():
        out1, _ = attention(x1, return_attention=True)
        out2, _ = attention(x2)

    assert not torch.allclose(out1, out2, atol=1e-4), \
        "Output is invariant to input changes — possible bug"


# =====================================================================
# 3. MASKING BEHAVIOR
# =====================================================================

def test_full_mask_equivalent_to_no_mask(dummy_input, attention, full_mask):
    """A mask of all ones should behave exactly like no mask."""
    attention.eval()
    with torch.no_grad():
        out_mask, attn_mask = attention(dummy_input, mask=full_mask, return_attention=True)
        out_none, attn_none = attention(dummy_input, return_attention=True)

    assert torch.allclose(out_mask, out_none, atol=1e-5), \
        "Full mask changed the output compared to no mask"
    assert torch.allclose(attn_mask, attn_none, atol=1e-5), \
        "Full mask changed attention weights compared to no mask"


def test_partial_mask_zeroes_masked_keys(dummy_input, attention, partial_mask):
    """Masked key positions must receive ~0 attention from every query."""
    _, attn_weights = attention(dummy_input, mask=partial_mask, return_attention=True)
    # attn_weights: (B, H, L, L) — last dim is keys
    assert torch.allclose(
        attn_weights[..., -1],
        torch.zeros_like(attn_weights[..., -1]),
        atol=1e-6
    ), "Masked key positions still receive attention"


def test_partial_mask_renormalizes(dummy_input, attention, partial_mask):
    """After masking, remaining weights must still sum to 1."""
    _, attn_weights = attention(dummy_input, mask=partial_mask, return_attention=True)
    sums = attn_weights.sum(dim=-1)
    expected = torch.ones_like(sums)
    assert torch.allclose(sums, expected, atol=1e-5), \
        "Masked attention weights do not renormalize to 1"


# =====================================================================
# 4. EDGE CASES & GUARDS
# =====================================================================

def test_embed_dim_not_divisible_by_num_heads():
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        MultiheadAttention(embed_dim=127, num_heads=8)


@pytest.mark.parametrize("bs,sl", [(1, 1), (1, 32), (8, 7), (4, 1)])
def test_various_batch_and_seq_lengths(bs, sl, embed_dim, num_heads, device):
    """Model must handle arbitrary batch sizes and sequence lengths."""
    attn = MultiheadAttention(embed_dim, num_heads).to(device)
    x = torch.randn(bs, sl, embed_dim, device=device)
    output, attn_weights = attn(x, return_attention=True)

    assert output.shape == (bs, sl, embed_dim)
    assert attn_weights.shape == (bs, num_heads, sl, sl)


def test_single_token_sequence(batch_size, embed_dim, num_heads, device):
    """With seq_len=1, the only token must attend to itself with weight 1."""
    attn = MultiheadAttention(embed_dim, num_heads).to(device)
    x = torch.randn(batch_size, 1, embed_dim, device=device)
    output, attn_weights = attn(x, return_attention=True)

    assert output.shape == (batch_size, 1, embed_dim)
    assert attn_weights.shape == (batch_size, num_heads, 1, 1)
    assert torch.allclose(attn_weights, torch.ones_like(attn_weights), atol=1e-5), \
        "Single-token attention weight is not 1.0"


def test_single_head(batch_size, seq_len, embed_dim, device):
    """Sanity check that num_heads=1 works."""
    attn = MultiheadAttention(embed_dim, num_heads=1).to(device)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    output, attn_weights = attn(x, return_attention=True)
    assert output.shape == x.shape
    assert attn_weights.shape == (batch_size, 1, seq_len, seq_len)


# =====================================================================
# 5. GRADIENT VERIFICATION
# =====================================================================

def test_backward_runs_without_error(dummy_input, attention):
    output, _ = attention(dummy_input, return_attention=True)
    loss = output.sum()
    loss.backward()  # should not raise


def test_all_parameters_receive_gradients(dummy_input, attention):
    output, _ = attention(dummy_input, return_attention=True)
    loss = output.sum()
    loss.backward()

    missing = []
    for name, param in attention.named_parameters():
        if param.grad is None:
            missing.append(name)
        else:
            assert not torch.isnan(param.grad).any(), \
                f"Gradient for {name} contains NaN"
            assert not torch.isinf(param.grad).any(), \
                f"Gradient for {name} contains Inf"

    assert len(missing) == 0, f"Parameters with no gradient: {missing}"


def test_input_receives_gradient(dummy_input, attention):
    x = dummy_input.clone().requires_grad_(True)
    output, _ = attention(x, return_attention=True)
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "Input did not receive gradient"
    assert x.grad.shape == x.shape


def test_gradient_with_mask(dummy_input, attention, partial_mask):
    """Gradients must flow correctly even when some positions are masked."""
    output, _ = attention(dummy_input, mask=partial_mask, return_attention=True)
    loss = output.sum()
    loss.backward()

    for name, param in attention.named_parameters():
        assert param.grad is not None, \
            f"Parameter {name} has no gradient when mask is used"


def test_gradient_magnitudes_are_reasonable(dummy_input, attention):
    """Catch vanishing or exploding gradients early."""
    output, _ = attention(dummy_input, return_attention=True)
    loss = output.sum()
    loss.backward()

    for name, param in attention.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            assert g_norm > 0, f"Zero gradient for {name}"
            assert g_norm < 1e6, \
                f"Exploding gradient for {name}: norm={g_norm:.2e}"


# =====================================================================
# 6. TRAINING MODE SANITY CHECKS
# =====================================================================

def test_train_mode_can_differ(dummy_input, attention):
    """Dropout (if present) should make two train-mode forwards differ."""
    attention.train()
    out1, _ = attention(dummy_input, return_attention=True)
    out2, _ = attention(dummy_input, return_attention=True)

    # Note: this is probabilistic; with p=0.1 dropout and 128-dim, P(equal) is negligible.
    # If your implementation has no dropout, change this to an `xfail` or remove it.
    if not torch.allclose(out1, out2, atol=1e-6):
        assert True  # Dropout is active
    else:
        pytest.skip("No dropout detected — train and eval produce identical outputs")