import torch 
import torch.nn as nn
import lightning as L

from src.models.modules import PositionalEncoding, TransformerEncoder

class TransformerPredictor(L.LightningModule):
    def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, dropout=0.0, input_dropout=0.0):
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

        self.cross_entropy = nn.CrossEntropyLoss()


    def _create_model(self):
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
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
        # x = self.input_net(x)
        # if add_positional_encoding:
        #     x = self.positional_encoding(x)
        # x = self.transformer(x, mask=mask)
        # x = self.output_net(x)
        # return x
        # x = self.input_net(x)
        # if add_positional_encoding:
        #     x = self.positional_encoding(x)
        # x = self.transformer(x, mask=mask)
        # x = x.mean(dim=1)           # pool over sequence
        # x = self.output_net(x)
        # return x
        x = self.input_net(x)                     # (B, L, model_dim)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)        # (B, L, model_dim)
        x = x.mean(dim=1)                         # (B, model_dim) pool here
        x = self.output_net(x)                    # (B, num_classes)
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
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError
    
    def test_step(self, batch, batch_idx):
        raise NotImplementedError


class SetAnomalyPredictor(TransformerPredictor):
    def forward(self, x, mask=None, add_positional_encoding=False):
        # Override to keep per‑element outputs (no pooling)
        x = self.input_net(x)                     # (B, L, model_dim)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)        # (B, L, model_dim)
        x = self.output_net(x)                    # (B, L, num_classes)
        return x
    
    def _calculate_loss(self, batch, mode="train"):
        # img_sets, _, labels = batch  # labels [B, seq_len]
        # preds = self(img_sets, add_positional_encoding=False) # [B, seq_len, num_classes]
        # preds = preds.squeeze(dim=-1)
        # loss = self.cross_entropy(preds, labels)
        # acc = (preds.argmax(dim=-1) == labels).float().mean()
        # self.log(f"{mode}_loss", loss)
        # self.log(f"{mode}_acc", acc)
        # return loss, acc
        img_sets, _, labels = batch   # labels: (batch, seq_len) with class indices (0 or 1)
        preds = self(img_sets, add_positional_encoding=False)  # (batch, seq_len, num_classes)
        batch_size, seq_len, num_classes = preds.shape
        preds = preds.view(-1, num_classes)      # (batch*seq_len, num_classes)
        labels = labels.view(-1)                 # (batch*seq_len)
        loss = self.cross_entropy(preds, labels)
        pred_classes = preds.argmax(dim=-1)      # (batch*seq_len)
        acc = (pred_classes == labels).float().mean()
        self.log(f"{mode}_loss", loss)
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
