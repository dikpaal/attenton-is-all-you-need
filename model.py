import math
import torch
import torch.nn as nn


class InputEnbeddings(nn.Module):

    def __init__(self, d: int, vocab_size: int):
        super().__init__()
        self.d = d
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(d)