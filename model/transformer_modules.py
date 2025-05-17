import math

import torch
import torch.nn as nn



# FAVORITE
class MultiHeadAttention(nn.Module):
    
    def __init__(self, d_model, h, dropout):
        
        """
        NOTES:
        
        - d_model is the mbedding vector dim
        - h is the number of heads
        - dk is the vector dim seen by each head
        - wq is W^Q
        - wk is W^K
        - wv is W^V
        - wo is W^O
        """
        
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dk = d_model // h
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        
        """
        NOTES:
        
        - ECAH OF QUERY, KEY, AND VALUE IS FROM (batch, len_of_sequence, d_model) TO (batch, len_of_sequence, d_model)
        """
        
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)


        query = query.view(query.shape[0], query.shape[1], self.h, self.dk).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.dk).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.dk).transpose(1, 2)
        # THE ABOVE: FROM (batch, len_of_sequence, d_model) --> (batch, len_of_sequence, h, dk) --> (batch, h, len_of_seuqnece, dk)
        

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        # COMBINE THE HEADS
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.dk) # FROM (batch, h, len_of_sequence, dk) TO (batch, len_of_sequence, h, dk) TO (batch, len_of_sequnece, d_model)

        return self.wo(x) # (batch, len_of_sequence, d_model) --> (batch, len_of_sequence, d_model)  
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        
        dk = query.shape[-1]
        
        attention_scores = ((query @ key.transpose(-2, -1)) / math.sqrt(dk)) # FROM (batch, h, len_of_sequence, dk) TO (batch, h, len_of_sequence, len_of_sequence)
        
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        attention_scores = attention_scores.softmax(dim = -1) # DIMENSION: (batch, h, len_of_sequence, len_of_sequence)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # DIM RED FROM (batch, h, len_of_sequence, len_of_sequence) TO (batch, h, len_of_sequence, dk)
        return (attention_scores @ value), attention_scores



class FeedForwardBlock(nn.Module):

    def __init__(self, d_model, dff, dropout):
        
        super().__init__()
        self.linear1 = nn.Linear(d_model, dff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dff, d_model)
    
    # FROM (batch,   len_of_sequence, d_model) TO (batch, len_of_sequence, dff) TO (batch, len_of_sequence, d_model)
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class PosEncoding(nn.Module):
    
    """
    MATHS:
    
    - CREATE MATRIX OF DIM (len_of_sequence, d_model)
    - CREATE VECTOR OF DIM len_of_sequence AND OF DIM d_model
    - CALC SIN OF EACH EVEN INDEX AND COS OF EACH ODD INDEX
    """

    def __init__(self, d_model, len_of_sequence, dropout):
                
        super().__init__()
        self.d_model = d_model
        self.len_of_sequence = len_of_sequence
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(len_of_sequence, d_model)
        
        position = torch.arange(0, len_of_sequence, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        # DIMENSION IS (batch, len_of_sequence, d_model)
        x += (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class NormLayer(nn.Module):

    def __init__(self, features, eps: float = 10 ** -6):
        
        super().__init__()
        self.eps = eps
        
        # APLHA
        self.alpha = nn.Parameter(torch.ones(features))
        
        # BIAS
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        
        
        # x: (batch, len_of_sequence, hidden_size)
        

        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
    def __init__(self, features, dropout):
        
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = NormLayer(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):

    def __init__(self, features, layers):
        
        super().__init__()
        self.layers = layers
        self.norm = NormLayer(features)

    def forward(self, x, mask):
    
        for layer in self.layers:
            x = layer(x, mask)
    
        return self.norm(x)
    
class Decoder(nn.Module):

    def __init__(self, features, layers):
        
        super().__init__()
        self.layers = layers
        self.norm = NormLayer(features)

    def forward(self, x, encoder_output, srcmask, tgtmask):
        
        for layer in self.layers:
            x = layer(x, encoder_output, srcmask, tgtmask)
        
        return self.norm(x)

class ProjLayer(nn.Module):

    def __init__(self, d_model, vocab_size):
        
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    # FROM (batch, len_of sequence, d_model) TO (batch, len_of_sequence, vocab_size)
    def forward(self, x):
        return self.proj(x)
    
class InputEmbeddings(nn.Module):

    def __init__(self, d_model, vocab_size):
        
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        
        return self.embedding(x) * math.sqrt(self.d_model)
