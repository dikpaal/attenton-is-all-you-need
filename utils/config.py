import torch
class Config:
    VOCAB_SIZE = 10
    len_of_sequence = 7
    BATCH_SIZE = 32
    EPOCHS = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-4