import numpy as np
import json
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import math
import argparse


MAX_LEN = 150
np.random.seed(422)


def parse_json(path: str) -> dict:
    with open(path, mode="r", encoding="UTF-8") as f:
        return json.load(f)


def softmax(value: np.array) -> np.array:
	exp = np.exp(value - np.max(value, axis=-1, keepdims=True))
	sum = np.sum(exp, axis=-1, keepdims=True)
	soft = exp / sum
	return soft


def GeLU(x: np.array) -> np.array:
    c = math.sqrt(2 / math.pi)
    inner = c * (x + 0.044715 * x**3)
    gelu  = 0.5 * x * (1 + np.tanh(inner))
    return gelu


def sigmoid(x: np.array) -> np.array:
    return 1.0 / (1.0 + math.e ** -x)  


def SiLU(x: np.array) -> np.array:
    return x * sigmoid(x)


def layer_norm(x: np.array, weight: np.array, bias: np.array, eps: float = 1e-5) -> np.array:
    mu   = x.mean(axis=-1, keepdims=True) # (seq, 1）
    var  = x.var(axis=-1, keepdims=True)  # (seq, 1)
    norm = (x - mu) / np.sqrt(var + eps)  # (seq, 768)
    return norm * weight + bias


def rms_norm(x: np.array, weight: np.array, eps: float = 1e-5) -> np.array:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight


def rope(x, theta):
    head_dim = x.shape[-1]
    seq_len = x.shape[1]

    inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2) / head_dim))

    angles = np.arange(seq_len)[:, None] * inv_freq[None, :]
    angles = np.concatenate((angles, angles), axis=-1)
    cos = np.cos(angles)[None, :, :]
    sin = np.sin(angles)[None, :, :]

    half = head_dim // 2
    rotated = np.concatenate((-x[..., half:], x[..., :half]),axis=-1)
    return x * cos + rotated * sin


def pred_next_tk(ids: list, W: dict, config: dict, temperature: float = 0.8) -> int:
    x = W["model.embed_tokens.weight"][ids]
    seqlen = x.shape[0]
    mask = np.triu(np.full((seqlen, seqlen), -np.inf), k = 1)

    for i in range(config["num_hidden_layers"]):
        residual = x
        h = rms_norm(x, W[f"model.layers.{i}.input_layernorm.weight"], eps=config["rms_norm_eps"])

        Q = h @ W[f"model.layers.{i}.self_attn.q_proj.weight"].T + W[f"model.layers.{i}.self_attn.q_proj.bias"]
        K = h @ W[f"model.layers.{i}.self_attn.k_proj.weight"].T + W[f"model.layers.{i}.self_attn.k_proj.bias"]
        V = h @ W[f"model.layers.{i}.self_attn.v_proj.weight"].T + W[f"model.layers.{i}.self_attn.v_proj.bias"]

        Q_heads = np.stack(np.split(Q, config["num_attention_heads"], axis=1), axis=0)
        K_heads = np.stack(np.split(K, config["num_key_value_heads"], axis=1), axis=0)
        V_heads = np.stack(np.split(V, config["num_key_value_heads"], axis=1), axis=0)

        attn_heads = []

        Q_heads = rope(Q_heads, config["rope_theta"])
        K_heads = rope(K_heads, config["rope_theta"])

        K_heads = np.repeat(K_heads, 7, axis=0)
        V_heads = np.repeat(V_heads, 7, axis=0)

        for j in range(config["num_attention_heads"]):
            hidden_dim = config["hidden_size"] // config["num_attention_heads"]
            score = Q_heads[j] @ K_heads[j].T / math.sqrt(hidden_dim)
            score = score + mask
            attn_head = softmax(score) @ V_heads[j] 
            attn_heads.append(attn_head)

        multi_head = np.concatenate(attn_heads, axis=-1)
        attn_out = multi_head @ W[f"model.layers.{i}.self_attn.o_proj.weight"].T
        x = residual + attn_out

        residual = x
        h = rms_norm(x, W[f"model.layers.{i}.post_attention_layernorm.weight"], eps=config["rms_norm_eps"])
        gate = SiLU(h @ W[f"model.layers.{i}.mlp.gate_proj.weight"].T)
        up = h @ W[f"model.layers.{i}.mlp.up_proj.weight"].T
        mid = gate * up
        out = mid @ W[f"model.layers.{i}.mlp.down_proj.weight"].T
        x = residual + out

    x = rms_norm(x, W["model.norm.weight"], config["rms_norm_eps"])
    x = x @ W["model.embed_tokens.weight"].T

    logits = x[-1]
    logits = logits / temperature
    probs = softmax(logits)

    next_id = np.random.choice(len(probs), p=probs)
    return next_id


def inference(context: str, weights: dict, config: dict, tokens: dict) -> None:
    print(context, end="", flush=True)
    
    ids = tokens.encode(context).ids

    while len(ids) < MAX_LEN:
        if MAX_LEN > 0 and len(ids) > MAX_LEN:
            break
        next_token = pred_next_tk(ids, weights, config)
        if next_token == config["eos_token_id"]:
            break
        ids.append(next_token)
        text = tokens.decode([next_token])
        print(text, end="", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    # Add a option to show model structure

    parser.add_argument("-mc", "--model_config", type=str, default="model/config.json")
    parser.add_argument("-t", "--tokenizer", type=str, default="model/tokenizer.json")
    parser.add_argument("-w", "--weights", type=str, default="model/model_fp32.safetensors")
    parser.add_argument("-c", "--context", type=str, default="""The following is a conversation between a User and a helpful Assistant.
       
    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant:""")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = parse_json(args.model_config)
    tokens = Tokenizer.from_file(args.tokenizer)
    weights = load_file(args.weights)

    keys = weights.keys()

    for k in keys:
        print(k, weights[k].shape)

    # return
    inference(args.context, weights, config, tokens)


if __name__ == "__main__":
    main()
