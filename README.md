# llminfer.np

A minimal LLM inference engine written from scratch in pure Python and NumPy.

Currently supports **GPT-2** and **Qwen2.5** families model. Inference without ML framework, every step from embeddings, multi head attention, MLP, sampling，and KV cache is implemented by hand. The engine support GPU acceleration using Cupy.

## Goal

The goal is to keep the code small and readable, then incrementally grow it toward supporting more model architectures.

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

For models support chat, multi turn chat mode can be activated.

```bash
uv run llminfer.py --chat
```

To change system prompt, use -c

```bash
uv run llminfer.py -c "Hello, my name is tom"
```


### Example

**Qwen2.5-0.5B-Instruct** chat mode.

```bash
uv run llminfer.py --chat -c "You are a helpful assistant, you role is assistant."

Assitant: Hi, i am an AI assistant.
User: What is your favorite city?        
Assistant: As an AI language model, I don't have personal preferences or emotions, so I don't have a favorite city. However, I can tell you about some popular cities around the world that people often enjoy visiting and are worth a visit.
User: Sure, please tell me
Assistant: Of course! Here are some cities that are popular worldwide:

1. New York City, USA
2. Tokyo, Japan
3. Paris, France
4. London, UK
5. Barcelona, Spain
6. Sydney, Australia
```


**Qwen2.5-0.5B-Instruct** chat mode off.

```bash
uv run llminfer.py

The capital of France is Paris. Paris is a beautiful historical city. It is a popular tourist destination and has many attractions. The city is known for its well-preserved old architecture, narrow streets and beautiful parks. It is also famous for its rich cultural heritage. Many visitors come to Paris for the food, especially the famous "Copé", served in the streets.
Paris is one of the most important cities in the world. It has a population of over 2 million people and is the capital of France, which has a population of over 6 million people. Paris is easily accessible by train, bus and car. It is also very convenient to travel to the nearby cities and countries, such as Lyon, Nice, Toulouse and Bordeaux.
```


**GPT-2 0.5B model output** only next token prediction, no chat mode

```bash
uv run llminfer.py --model gpt2

The capital of France is Paris. Paris is a beautiful historical city. It is a modern industrial city with many notable museums and more than 20 million people living in it.

The biggest part of that is the French capital. There are 11 million inhabitants in France. At the same time it is the biggest city in the world with over 30 billion people and there are numerous museums and other tourist attractions.

The city is interesting because it is one of the most beautiful cities in the world. If you are thinking about how much you could spend on French food, Paris is probably the best place to start.

If you are looking for a country to get into, France is a great choice but you will have to look elsewhere. If you ...
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
