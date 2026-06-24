#!/bin/bash

# Download GPT-2 weights from Hugging Face.
# Only 3 files are needed by the engine:
#   model.safetensors  - weights
#   config.json        - architecture (n_layer / n_head / n_embd ...)
#   tokenizer.json     - BPE tokenizer


set -e

FILES="model.safetensors config.json tokenizer.json"

# # --- gpt2 (124M) -------------------------------------------------------------
hf download openai-community/gpt2 $FILES --local-dir ./model

# --- gpt2-medium (355M) ------------------------------------------------------
# hf download openai-community/gpt2-medium $FILES --local-dir ./model-medium

# --- gpt2-large (774M) -------------------------------------------------------
# hf download openai-community/gpt2-large $FILES --local-dir ./model-large

# --- gpt2-xl (1.5B) ----------------------------------------------------------
# hf download openai-community/gpt2-xl $FILES --local-dir ./model-xl

echo "Model has been downloaded."