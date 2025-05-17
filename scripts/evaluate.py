import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model.transformer import init_transformer
from utils.masks import create_mask
from utils.config import Config

model = init_transformer(
    src_vocab_size=Config.VOCAB_SIZE,
    tgt_vocab_size=Config.VOCAB_SIZE,
    src_len_of_sequence=Config.SEQ_LEN,
    tgt_len_of_sequence=Config.SEQ_LEN
)

model.load_state_dict(torch.load("checkpoints/transformer.pt"))
model.eval()
model.to(Config.DEVICE)

src_sequence = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], dtype=torch.long).to(Config.DEVICE)
src_mask, _ = create_mask(src_sequence, src_sequence)

# Start with the first token as input
generated = torch.tensor([[1]], dtype=torch.long).to(Config.DEVICE)

# Generate tokens one by one
for _ in range(Config.SEQ_LEN - 1):
    _, tgt_mask = create_mask(src_sequence, generated)
    enc_out = model.encode(src_sequence, src_mask)
    dec_out = model.decode(enc_out, src_mask, generated, tgt_mask)
    output = model.project(dec_out)
    next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated = torch.cat((generated, next_token), dim=1)

print("Input sequence:     ", src_sequence[0].tolist())
print("Generated sequence: ", generated[0].tolist())
