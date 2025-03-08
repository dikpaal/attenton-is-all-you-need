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
    
class PositionalEncoding(nn.Module):

    def __init__(self, d: int, sequence_len: int, dropout: float) -> None:
        
        super().__init__()
        self.d = d
        self.sequence_len = sequence_len
        self.dropout = dropout

        pe = torch.zeros(sequence_len, d) # Matrix of shape (sequence_len, d)
        
        # Vector of shape (sequence_len, 1)
        pos = torch.arange(0, sequence_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (- math.log(10000.0) / d))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x += (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
    
        # x: (batch, sequence_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, sequence_len, 1)
        
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, sequence_len, 1)
        
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1, b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2, b2

    def forward(self, x):
        
        # (batch, sequence_len, d_model) --> (batch, sequence_len, d_ff) --> (batch, sequence_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))