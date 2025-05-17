# 🧠 Attention Is All You Need — Transformer from Scratch

A full PyTorch implementation of the Transformer architecture proposed in the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper by Vaswani et al., built **entirely from scratch** — no `nn.Transformer` used. This repo walks through **core components** like Multi-Head Attention, Positional Encoding, Residual Connections, and builds a full encoder-decoder architecture.

---

## 📌 Table of Contents

- [📖 Background](#-background)
- [🛠 Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🚀 How to Run](#-how-to-run)
- [📈 Training Phase](#-training-phase)
- [📊 Results](#-results)
- [📚 References](#-references)

---

## 📖 Background

> “Attention is all you need” introduced a novel, fully attention-based architecture that eliminated recurrence entirely in sequence transduction tasks.

This project recreates that model step by step using only **PyTorch base modules**.

Key ideas:
- Multi-Head Scaled Dot-Product Attention
- Positional Encoding (sinusoidal)
- Layer Normalization & Residual Connections
- Encoder-Decoder Structure

---

## 🛠 Features

✅ Pure PyTorch implementation (no high-level `nn.Transformer`)  
✅ Modular, extensible architecture  
✅ Toy `CopyDataset` for sanity checking  
✅ Supports batch training, masking, and auto-regressive decoding  
✅ Well-structured for later expansion to translation, summarization, etc.

---

## 📂 Project Structure

```bash
transformer_project/
├── data/
│   └── dataset.py              # Copy task dataset
├── model/
│   ├── transformer.py          # Transformer architecture
│   └── transformer_modules.py  # MHA, FFN, PosEnc, etc.
├── train/
│   └── train_loop.py           # Training logic (if separated)
├── utils/
│   ├── config.py               # Hyperparameters
│   └── masks.py                # Attention masks
├── scripts/
│   ├── train.py                # Run training
│   └── evaluate.py             # (Coming soon) Test predictions
└── README.md
```

---

## 🚀 How to Run

### ✅ 1. Set up environment

```bash
conda create -n attention-fixed python=3.9
conda activate attention-fixed
pip install torch numpy
```

### ✅ 2. Train the model

```bash
python scripts/train.py
```

### ✅ 3. Evaluate on custom inputs *(after `evaluate.py` is added)*

---

## 📈 Training Phase

The model is trained on a simple **CopyDataset** — a synthetic task where the target is to exactly match the input sequence. This is a standard sanity check for sequence-to-sequence models.

The loss over 10 epochs is shown below:

```
Epoch 1: Loss = 65.6407
Epoch 2: Loss = 45.7510
Epoch 3: Loss = 41.2413
Epoch 4: Loss = 38.0328
Epoch 5: Loss = 35.7573
Epoch 6: Loss = 31.8876
Epoch 7: Loss = 24.8849
Epoch 8: Loss = 18.8855
Epoch 9: Loss = 14.8475
Epoch 10: Loss = 11.7760
```

This decreasing trend in cross-entropy loss demonstrates that the model is successfully learning the identity function, i.e., copying the input sequence to the output.

---

## 📊 Results

Sample loss curve on CopyDataset (vocab size = 10, seq len = 7):

| Epoch | Loss     |
|-------|----------|
| 1     | 65.64    |
| 5     | 35.75    |
| 10    | 11.77    |

👉 Model is able to reliably learn the identity function (copying inputs).

---

## 📚 References

- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer (Harvard NLP)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
