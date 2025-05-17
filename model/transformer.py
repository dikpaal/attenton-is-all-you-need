import torch.nn as nn
from model.transformer_modules import MultiHeadAttention, FeedForwardBlock, PosEncoding, ResidualConnection, Encoder, Decoder, ProjLayer, InputEmbeddings


class EncoderBlock(nn.Module):

    def __init__(self, features, self_attention_block, feed_forward_block, dropout):
        
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, srcmask):
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, srcmask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x

class DecoderBlock(nn.Module):

    def __init__(self, features, self_attention_block, cross_attention_block, feed_forward_block, dropout):
    
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, srcmask, tgtmask):
    
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgtmask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, srcmask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x
    
class Transformer(nn.Module):

    def __init__(self, encoder, decoder, srcembed, tgtembed, src_pos, tgtpos, projection_layer):
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srcembed = srcembed
        self.tgtembed = tgtembed
        self.src_pos = src_pos
        self.tgtpos = tgtpos
        self.projection_layer = projection_layer


    # DIMENSION: (batch, len_of_sequence, d_model)
    def encode(self, src, srcmask):
    
        src = self.srcembed(src)
        src = self.src_pos(src)
        return self.encoder(src, srcmask)
    
    # DIMESNION (batch, len_of_sequence, d_model)
    def decode(self, encoder_output, srcmask, tgt, tgtmask):
    
        tgt = self.tgtembed(tgt)
        tgt = self.tgtpos(tgt)
        return self.decoder(tgt, encoder_output, srcmask, tgtmask)
    
    # DIMENSION: (batch, len_of_sequence, vocab_size)
    def project(self, x):
        
        return self.projection_layer(x)
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        encoded = self.encode(src, src_mask)
        decoded = self.decode(encoded, src_mask, tgt, tgt_mask)
        return self.project(decoded)
    

def init_transformer(src_vocab_size, tgt_vocab_size, src_len_of_sequence, tgt_len_of_sequence, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, dff: int = 2048):
    
    """
    PROGRESS:
    
    - EMBEDDING LAYERS: DONE
    - POSITIONAL ENCODING LATERS: DONE
    - ENCODER BLOCKS: DONE
    - DECODER BLOCKS: DONE
    - ENCODER: DONE
    - DECODER: DONE
    - PROJECTION LAYER: DONE
    - INIT TRANSFORMER
    - INIT PARAMS
    """
    
    srcembed = InputEmbeddings(d_model, src_vocab_size)
    tgtembed = InputEmbeddings(d_model, tgt_vocab_size)
    src_pos = PosEncoding(d_model, src_len_of_sequence, dropout)
    tgtpos = PosEncoding(d_model, tgt_len_of_sequence, dropout)
    
    # ENCODER
    encoder_blocks = []
    
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # DECODER
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, dff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    projection_layer = ProjLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, srcembed, tgtembed, src_pos, tgtpos, projection_layer)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

