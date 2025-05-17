import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random


# Dummy dataset where input == target
class CopyDataset(Dataset):
    def __init__(self, vocab_size=10, len_of_sequence=7, num_samples=1000):
        self.vocab_size = vocab_size
        self.len_of_sequence = len_of_sequence
        self.samples = [
            torch.randint(1, vocab_size, (len_of_sequence,))
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        y = x.clone()
        return x, y