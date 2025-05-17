import torch
class Config:
    VOCAB_SIZE = 10
    SEQ_LEN = 7
    BATCH_SIZE = 32
    EPOCHS = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-4