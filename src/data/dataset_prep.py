import torch
from torch.utils.data import Dataset

import numpy as np


class SetAnomalyDataset(Dataset):
    def __init__(self, num_sets, set_size, num_classes, img_dim, anomaly_label=None):
        self.num_sets = num_sets
        self.set_size = set_size
        self.num_classes = num_classes
        self.img_dim = img_dim
        self.anomaly_label = anomaly_label

        self.data = []
        self.labels = []

        for _ in range(num_sets):
            set_data, set_labels = self._generate_set()
            self.data.append(set_data)
            self.labels.append(set_labels)
    def __len__(self):
        return self.num_sets

    def __getitem__(self, idx):
        # Return (set_data, mask, target)
        # mask is all ones because we don't have any padding in the set
        return self.data[idx], torch.ones(self.set_size), self.labels[idx]

    def _generate_set(self):
        main_label = np.random.randint(self.num_classes)

        if self.anomaly_label is None:
            anomaly_label = np.random.randint(self.num_classes)
            while anomaly_label == main_label:
                anomaly_label = np.random.randint(self.num_classes)
        else:
            anomaly_label = self.anomaly_label

        anomaly_pos = np.random.randint(self.set_size)

        class_means = {
        main_label: torch.randn(self.img_dim) * 2.0,
        anomaly_label: torch.randn(self.img_dim) * 2.0,
    }
        
        set_data = []
        set_labels = []

        for i in range(self.set_size):
            if i == anomaly_pos:
                label = anomaly_label
                
            else:
                label = main_label
            
            # Add Gaussian noise around the class mean
            element = class_means[label] + torch.randn(self.img_dim)
            set_data.append(element)
            set_labels.append(label)

        # Convert to tensors
        set_data = torch.stack(set_data)                     # (set_size, img_dim)
        set_labels = torch.tensor(set_labels, dtype=torch.long)  # (set_size,)

        # Binary target: 1 for anomaly, 0 for normal
        target = (set_labels == anomaly_label).long()        # (set_size,)

        return set_data, target
