# llminfer.np

A minimal LLM inference engine written from scratch in pure Python and NumPy.

Currently supports **GPT-2** families model (from 124M to 1.5B). Without PyTorch, ML framework, every step from
embeddings, attention, MLP, sampling，and KV cache is implemented by hand.

## Goal

To understand LLM inference end-to-end by building it. The goal
is to keep the code small and readable, then incrementally grow it toward supporting more model
architectures.

## Features

- Pure NumPy forward pass — no deep learning framework
- Loads official weights via `safetensors`
- Adapt variety of GPT-2 family model structure with official configuration file
- KV cache for acceleration of model inference
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
# Inference without KV cache
python inference.py

# Inference with KV cache
python inference_KV.py
```

### Example
```
The following is a conversation between a User and a helpful Assistant.

    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant: The capital of France is Paris.


User: I'm using it to register a new account. Do you want to invite me to your account?

Assistant: You don't have to invite me to the account, but you do have to do something that you normally would not do with someone else. I'll be at the account.


User: Is your account private?

Assistant: If you're using my name and password, or your (...)
```

To change prompt, use -c

```bash
python inference.py -c "Hello, my name is tom"
```

### KV cache accelaration

For 150 tokens generation:

- **python inference_KV.py** 
   - 95.20s user 59.38s system 933% cpu 16.561 total 
- **python inference.py**  
   - 356.33s user 78.85s system 952% cpu 45.688 total



## How it works

input text
- → BPE tokenizer            (tokenizers)
- → token + position embeddings
- → n × transformer block
        ln_1 → multi-head attention → residual
        ln_2 → MLP (GELU)           → residual
- → final layer norm
- → logits = x @ wte.T            (tied embedding)
- → temperature sampling
- → decode → next token
- → loop

Roadmap

- [ x ] KV cache (avoid recomputing the full sequence each step)
- [ ] Add GPU support
- [ ] top-k / top-p sampling
- [ ] Support more architectures (RoPE, LLaMA / Qwen)
- [ ] Hand-written BPE tokenizer

# Notes

GPT-2 is a base model (no chat/instruction fine-tuning) — it continues text rather than answering.
Repetition with greedy decoding is expected; use temperature sampling for variety.


## Acknowledgments & References

  This project is a learning exercise, inspired by and indebted to:

  - **Andrej Karpathy** — for making transformers approachable to everyone:
    - [nanoGPT](https://github.com/karpathy/nanoGPT) — minimal GPT training/inference in PyTorch
    - [minGPT](https://github.com/karpathy/minGPT) — the earlier minimal GPT
    - ["Let's reproduce GPT-2"](https://www.youtube.com/watch?v=l8pRSuU81PU)
  - **Jay Mody** — [picoGPT](https://github.com/jaymody/picoGPT), GPT-2 forward pass in ~60 lines of
  NumPy
  - **OpenAI** — [GPT-2](https://github.com/openai/gpt-2) and the
    [original paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_mu
  ltitask_learners.pdf)
  - **Hugging Face** — model weights (`openai-community/gpt2`), `safetensors`, and `tokenizers`


# License

MIT
