import torch

def create_mask(src, tgt, pad_token=0):
    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
    tgt_pad_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
    
    seq_len = tgt.size(1)
    tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()
    
    tgt_mask = tgt_pad_mask & tgt_sub_mask  # (batch, 1, tgt_len, tgt_len)
    return src_mask, tgt_mask