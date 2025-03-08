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

    def __init__(self, d: int, d_ff: int, dropout: float) -> None:
        
        super().__init__()
        self.linear_1 = nn.Linear(d, d_ff) # w1, b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d) # w2, b2

    def forward(self, x):
        
        # (batch, sequence_len, d) --> (batch, sequence_len, d_ff) --> (batch, sequence_len, d)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d: int, h: int, dropout: float) -> None:
        
        super().__init__()
        self.d = d
        self.h = h
        assert d % h == 0, "d is not divisible by h" # d should be divisible by h

        self.d_k = d // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
    
        d_k = query.shape[-1]
        # (batch, h, sequence_len, d_k) --> (batch, h, sequence_len, sequence_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
    
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, sequence_len, sequence_len) # Apply softmax
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
    
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
       
        query = self.w_q(q) # (batch, sequence_len, d) --> (batch, sequence_len, d)
        key = self.w_k(k) # (batch, sequence_len, d) --> (batch, sequence_len, d)
        value = self.w_v(v) # (batch, sequence_len, d) --> (batch, sequence_len, d)

        # (batch, sequence_len, d) --> (batch, sequence_len, h, d_k) --> (batch, h, sequence_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, sequence_len, d_k) --> (batch, sequence_len, h, d_k) --> (batch, sequence_len, d)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, sequence_len, d) --> (batch, sequence_len, d)  
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:

            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):

            return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:

        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

