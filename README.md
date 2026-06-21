# llminfer.np

A minimal LLM inference engine written from scratch in pure Python and NumPy.

Currently supports **GPT-2** families model (from 124M to 1.5B). No PyTorch, no ML framework, every step from
embeddings, attention, MLP, sampling is implemented by hand.

## Goal

To understand LLM inference end-to-end by building it. The goal
is to keep the code small and readable, then incrementally grow it toward supporting more model
architectures.

## Features

- Pure NumPy forward pass — no deep learning framework
- Loads official weights via `safetensors`
- Full GPT-2 transformer block: pre-norm, multi-head causal self-attention, GELU MLP
- Temperature sampling with EOS stopping
- Streaming token-by-token output

## Usage

### Setup & model preparation

```bash
# Set up virtual environment
bash venv.sh

# Install dependencies
pip install -r requirements.txt

# Enter virtual environment
source venv/bin/activate

# Download the GPT-2 weights and config into a model/ folder
bash download_model.sh

# By default, 124m model will be downloaded, but can be modified in download_model.sh
```

### Usage
```bash
python inference.py
```

Edit the prompt in main():

chat = "Hello, my name is tom"

How it works

text
- → BPE tokenizer            (tokenizers)
- → token + position embeddings
- → 12 × transformer block
        ln_1 → multi-head attention → residual
        ln_2 → MLP (GELU)           → residual
- → final layer norm
- → logits = x @ wte.T            (tied embedding)
- → temperature sampling
- → decode → next token
- → loop

Roadmap

- [ ] KV cache (avoid recomputing the full sequence each step)
- [ ] Add GPU support
- [ ] top-k / top-p sampling
- [ ] Support more architectures (RoPE, LLaMA / Qwen)
- [ ] Hand-written BPE tokenizer

# Notes

GPT-2 is a base model (no chat/instruction fine-tuning) — it continues text rather than answering.
Repetition with greedy decoding is expected; use temperature sampling for variety.

# License

MIT
