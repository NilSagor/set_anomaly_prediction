import torch 
import torch.nn as nn
import lightning as L

from src.models.modules import PositionalEncoding, TransformerEncoder

class TransformerPredictor(L.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, dropout=0.0, input_dropout=0.0):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

        self.cross_entropy = nn.CorssEntropy()


    def _create_model(self):
        self.input_net = nn.Sequential(
            nn.Dropout(self.hprams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )

        self.positional_encoding = PositionalEncoding(
            d_model = self.hparams.model_dim
        )

        self.transformer = TransformerEncoder(
            num_layers = self.hparams.num_layers,
            input_dim = self.hparams.model_dim,
            dim_feedforward = 2*self.hparams.model_dim,
            num_heads = self.hparams.num_heads,
            dropout = self.hparams.dropout
        )
        
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        x = self.output_net(x)
        return x
    
    @torch.no_grad
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        raise NotImplemented
    
    def validation_step(self, batch, batch_idx):
        raise NotImplemented
    
    def test_step(self, batch, batch_idx):
        raise NotImplemented


class SetAnomalyPredictor(TransformerPredictor):
    def _calculate_loss(self, batch, mode="train"):
        img_sets, _, labels = batch 
        preds = self(img_sets, add_positional_encoding=False)
        preds = preds.squeeze(dim=-1)
        loss = self.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
