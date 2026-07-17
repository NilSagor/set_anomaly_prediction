import pytest
import torch
import lightning as L

# ============================================================
# TransformerPredictor
# ============================================================

class TestTransformerPredictor:
    def test_is_lightning_module(self, transformer_predictor):
        assert isinstance(transformer_predictor, L.LightningModule)

    def test_has_required_modules(self, transformer_predictor):
        assert hasattr(transformer_predictor, "input_net")
        assert hasattr(transformer_predictor, "positional_encoding")
        assert hasattr(transformer_predictor, "transformer")
        assert hasattr(transformer_predictor, "output_net")

    def test_forward_shape(self, dummy_input, transformer_predictor):
        transformer_predictor.eval()
        with torch.no_grad():
            out = transformer_predictor(dummy_input, add_positional_encoding=True)
        assert out.shape == (dummy_input.shape[0], transformer_predictor.hparams.num_classes)
    
    
    def test_forward_without_positional_encoding(self, dummy_input, transformer_predictor):
        transformer_predictor.eval()
        with torch.no_grad():
            out = transformer_predictor(dummy_input, add_positional_encoding=False)
        assert out.shape == (dummy_input.shape[0], transformer_predictor.hparams.num_classes)


    def test_forward_with_mask(self, dummy_input, transformer_predictor):
        mask = torch.ones(dummy_input.shape[0], dummy_input.shape[1], device=dummy_input.device)
        transformer_predictor.eval()
        with torch.no_grad():
            out_mask = transformer_predictor(dummy_input, mask=mask)
            out_none = transformer_predictor(dummy_input, mask=None)
        assert out_mask.shape == out_none.shape

    def test_forward_preserves_device(self, dummy_input, transformer_predictor, device):
        transformer_predictor.eval()
        with torch.no_grad():
            out = transformer_predictor(dummy_input)
        assert out.device == device

    def test_get_attention_maps_shape(self, dummy_input, transformer_predictor, num_heads):
        transformer_predictor.eval()
        attn_maps = transformer_predictor.get_attention_maps(dummy_input)
        b, l, _ = dummy_input.shape
        num_layers = transformer_predictor.hparams.num_layers
        assert len(attn_maps) == num_layers
        for attn in attn_maps:
            assert attn.shape == (b, num_heads, l, l)

    def test_get_attention_maps_no_grad(self, dummy_input, transformer_predictor):
        """Ensure get_attention_maps does not build a computation graph."""
        attn_maps = transformer_predictor.get_attention_maps(dummy_input)
        for attn in attn_maps:
            assert attn.grad_fn is None, "Attention map has grad_fn — no_grad not applied"


    def test_configure_optimizers(self, transformer_predictor):
        optim = transformer_predictor.configure_optimizers()
        assert isinstance(optim, torch.optim.Adam)

    def test_hyperparameters_saved(self, transformer_predictor):
        assert transformer_predictor.hparams.lr == 1e-3
        assert transformer_predictor.hparams.num_layers == 2

    def test_gradient_flow(self, dummy_input, transformer_predictor):
        """Verify backward passes through the full model."""
        transformer_predictor.train()
        out = transformer_predictor(dummy_input)
        loss = out.sum()
        loss.backward()

        for name, param in transformer_predictor.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"



class TestSetAnomalyPredictor:
    def test_forward_shape(self, dummy_input, set_anomaly_predictor):
        set_anomaly_predictor.eval()
        with torch.no_grad():
            out = set_anomaly_predictor(dummy_input, add_positional_encoding=False)
        expected_shape = (dummy_input.shape[0], dummy_input.shape[1])
        assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"

    def test_forward_with_positional_encoding(self, dummy_input, set_anomaly_predictor):
        set_anomaly_predictor.eval()
        with torch.no_grad():
            out = set_anomaly_predictor(dummy_input, add_positional_encoding=True)
        expected_shape = (dummy_input.shape[0], dummy_input.shape[1])
        assert out.shape == expected_shape

    def test_training_step_runs(self, set_anomaly_predictor, dummy_batch):
        set_anomaly_predictor.train()
        loss = set_anomaly_predictor.training_step(dummy_batch, batch_idx=0)
        assert loss is not None
        assert torch.isfinite(loss)

    def test_validation_step_runs(self, set_anomaly_predictor, dummy_batch):
        # No assertion need– no exception 
        set_anomaly_predictor.eval()
        with torch.no_grad():
            set_anomaly_predictor.validation_step(dummy_batch, batch_idx=0)
        

    def test_test_step_runs(self, set_anomaly_predictor, dummy_batch):
        set_anomaly_predictor.eval()
        with torch.no_grad():
            set_anomaly_predictor.test_step(dummy_batch, batch_idx=0)

    def test_loss_calculation(self, set_anomaly_predictor, dummy_batch):
        set_anomaly_predictor.train()
        loss, acc = set_anomaly_predictor._calculate_loss(dummy_batch, mode="train")
        assert torch.isfinite(loss)
        assert 0.0 <= acc <= 1.0

    def test_gradient_flow(self, dummy_input, set_anomaly_predictor):
        set_anomaly_predictor.train()
        out = set_anomaly_predictor(dummy_input)
        loss = out.sum()
        loss.backward()
        for name, param in set_anomaly_predictor.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"