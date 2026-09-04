# llminfer.np

A minimal LLM inference engine written from scratch in pure Python and NumPy.

Currently supports **GPT-2** and **Qwen2.5** families model. Inference without ML framework, every step from
embeddings, attention, MLP, sampling，and KV cache is implemented by hand. The engine support GPU acceleration using Cupy.

## Goal

The goal is to keep the code small and readable, then incrementally grow it toward supporting more model
architectures.

This repository contains 2 implementations:
  - **llminfer_minimal.py** - The minimalist runable version, implement basic GPT-2 transformer decoder, the code is more readable, supporting CUDA and KV cache, allowing faster inference speed..

  - **llminfer.py** - The version supports more model architectures, and can switch model through arguments.

Two versions are for different purpose.

## Features
- Pure NumPy / Cupy forward pass, with only matrix operation — no deep learning framework
- Loads official weights via `safetensors`
- KV cache for acceleration of model inference
- Full GPT-2 transformer block: pre-norm, multi-head self-attention, GELU MLP
- Modern architecture improvement with Qwen2.5, RMSNorm, SwigGLU, ROPE
- Temperature sampling with EOS stopping

## Usage

### Environment

MacOS is supported with CPU inference only. CUDA acceleration is
available on Linux with CUDA 12.x.

uv is used to manage virtual environment.

### Setup & model preparation

```bash
uv sync
```

For CUDA
```bash
uv sync --extra cuda
```

```bash
# Download the GPT-2 and Qwen2.5 weights and config into a model/ folder
bash download_model.sh

# By default, GPT2 124m model will be downloaded, but can be modified in download_model.sh
```

### Usage
```bash
# Inference without KV cache, by default, it use Qwen2.5
uv run llminfer.py

# Inference with KV cache
uv run llminfer.py -kv

# Inference with CUDA
uv run llminfer.py -cu

# Inference with KV cache and CUDA
uv run llminfer.py -kv -cu
```

### Example

**GPT-2 1.5B model output**

```
Paris is one of the most touristic city, here are the best places to enjoy your trip.

Find an old friend in Paris

On the day of your trip, make sure that you visit the famous Champs de Mars, a collection of monuments founded by Napoleon Bonaparte in 1811. The Champs de Mars are a must see for any Parisian.

A huge collection of monuments that were inaugurated during the reign of Napoleon Bonaparte, the Champs de Mars are a must-see for any Parisian. The buildings are recognizable by their giant arches, and the fountain-beds are covered in marble

The Champs de Mars are a must see for any Parisian, no matter if ...
```

To change prompt, use -c

```bash
uv run llminfer.py -c "Hello, my name is tom"
```

### Accelaration

For 150 total tokens:

Device 
CPU  i7 13650HX
GPU  RTX 4060

#### Without acceleration
- [CPU with no acceleration]
  - **python llminfer_minimal.py** 1073.20s user 17.07s system 1313% cpu 1:23.02 total
- [KV cache] 
  - **python llminfer_minimal.py -kv**  426.58s user 4.88s system 839% cpu 51.372 total
- [CUDA]
  - **python llminfer_minimal.py -cu**  15.17s user 2.20s system 104% cpu 16.672 total
- [KV cache + CUDA]
  - **python llminfer_minimal.py -kv -cu**  6.22s user 1.08s system 120% cpu 6.049 total

As we can observe, GPU acceleration increase 5 times of inference speed than CPU, KV cache increase 1.6 times.


## Future roadmap

- [ x ] KV cache (avoid recomputing the full sequence each step)
- [ x ] Add GPU support
- [ ] top-k / top-p sampling
- [ x ] Support more architectures (RoPE, LLaMA / Qwen)
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
    [original paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)


# License

MIT
