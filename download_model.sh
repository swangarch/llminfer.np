#!/bin/bash

# Download GPT-2 weights from Hugging Face.
# Only 3 files are needed by the engine:
#   model.safetensors  - weights
#   config.json        - architecture (n_layer / n_head / n_embd ...)
#   tokenizer.json     - BPE tokenizer


set -e

FILES="model.safetensors config.json tokenizer.json"

# --- gpt2 (124M) -------------------------------------------------------------
hf download openai-community/gpt2 $FILES --local-dir ./model/gpt2

# --- gpt2-medium (355M) ------------------------------------------------------
# hf download openai-community/gpt2-medium $FILES --local-dir ./model/gpt2

# --- gpt2-large (774M) -------------------------------------------------------
# hf download openai-community/gpt2-large $FILES --local-dir ./model/gpt2

# --- gpt2-xl (1.5B) ----------------------------------------------------------
# hf download openai-community/gpt2-xl $FILES --local-dir ./model/gpt2


# # --- Qwen 2.5 -------------------------------------------------------------
hf download Qwen/Qwen2.5-0.5B-Instruct --local-dir ./model/qwen2.5



echo "Model has been downloaded."