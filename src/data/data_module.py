import torch
import lightning as L
from torch.utils.data import DataLoader, random_split

from .dataset_prep import SetAnomalyDataset

class SetAnomalyDataModule(L.LightningDataModule):
    def __init__(self, set_size:int = 10, num_classes:int=10, img_dim:int=128, batch_size:int=32, num_train_sets:int=1000, num_val_sets:int=100, num_test_sets:int=100, num_workers:int=4, anomaly_labels:int=None):
        super().__init__()
        self.save_hyperparameters(ignore=['num_classes', 'anomaly_label'])
        self.set_size = set_size
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.batch_size = batch_size
        self.num_train_sets = num_train_sets
        self.num_val_sets = num_val_sets
        self.num_test_sets = num_test_sets
        self.num_workers = num_workers
        self.anomaly_labels = anomaly_labels

    def setup(self, stage=None):
        full_dataset = SetAnomalyDataset(
            num_sets=self.num_train_sets + self.num_val_sets + self.num_test_sets,
            set_size=self.set_size,
            num_classes=self.num_classes,
            img_dim=self.img_dim,
            anomaly_label=self.anomaly_labels,
        )
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [self.num_train_sets, self.num_val_sets, self.num_test_sets],
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
