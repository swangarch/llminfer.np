import numpy as np
import json
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import math
import argparse


MAX_LEN = 150
np.random.seed(422)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-mc", "--model_config", type=str, default="model/config.json")
    parser.add_argument("-t", "--tokenizer", type=str, default="model/tokenizer.json")
    parser.add_argument("-w", "--weights", type=str, default="model/model.safetensors")
    parser.add_argument("-c", "--context", type=str, default="""The following is a conversation between a User and a helpful Assistant.

    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant:""")

    args = parser.parse_args()
    return args


def parse_json(path: str) -> dict:
    try:
        with open(path, mode="r", encoding="UTF-8") as f:
            content = json.load(f)
            return content
    except Exception as e:
        print("Error:", e)
        return None


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


def layer_norm(x: np.array, weight: np.array, bias: np.array, eps: float = 1e-5) -> np.array:
    mu   = x.mean(axis=-1, keepdims=True) # (seq, 1）
    var  = x.var(axis=-1, keepdims=True)  # (seq, 1)
    norm = (x - mu) / np.sqrt(var + eps)  # (seq, 768)
    return norm * weight + bias


def pred_next_tk(ids: list, W: np.array, config: dict, kv_cache: list,
                 prefil: bool = False, temperature: float = 0.8) -> int:
    if prefil:
        x = W["wte.weight"][ids] + W["wpe.weight"][np.arange(len(ids))]
    else:
        pos = len(ids) - 1
        x = W["wte.weight"][[ids[-1]]] + W["wpe.weight"][[pos]]
    seqlen = x.shape[0]
    mask = np.triu(np.full((seqlen, seqlen), -np.inf), k=1)

    for i in range(config["n_layer"]):
        x1 = layer_norm(x, W[f"h.{i}.ln_1.weight"], W[f"h.{i}.ln_1.bias"], config["layer_norm_epsilon"])
        if prefil:
            QKV = x1 @ W[f"h.{i}.attn.c_attn.weight"] + W[f"h.{i}.attn.c_attn.bias"]
            Q, K, V = np.split(QKV, 3, axis=-1)
        else:
            Wq, Wk, Wv = np.split(W[f"h.{i}.attn.c_attn.weight"], 3, axis=-1)
            Bq, Bk, Bv = np.split(W[f"h.{i}.attn.c_attn.bias"], 3, axis=-1)
            Q = x1 @ Wq + Bq
            K = np.concatenate([kv_cache[i]["K"],  (x1[-1, :] @ Wk + Bk).reshape(1, -1)], axis=0)
            V = np.concatenate([kv_cache[i]["V"],  (x1[-1, :] @ Wv + Bv).reshape(1, -1)], axis=0)

        if prefil:
            kv_cache.append({"K": K, "V": V})
        else:   
            kv_cache[i] = {"K": K, "V": V}
        Q_heads = np.split(Q, config["n_head"], axis=-1)
        K_heads = np.split(K, config["n_head"], axis=-1)
        V_heads = np.split(V, config["n_head"], axis=-1)

        attn_heads = []
        for j in range(config["n_head"]):
            score = Q_heads[j] @ K_heads[j].T / (math.sqrt(K_heads[j].shape[1]))
            if prefil:
                score = score + mask
            attn_head = softmax(score) @ V_heads[j]
            attn_heads.append(attn_head)
        
        multi_head = np.concatenate(attn_heads, axis=-1)
        x1 = multi_head @ W[f"h.{i}.attn.c_proj.weight"] + W[f"h.{i}.attn.c_proj.bias"]
        x = x + x1 # res

        x2 = layer_norm(x, W[f"h.{i}.ln_2.weight"], W[f"h.{i}.ln_2.bias"], config["layer_norm_epsilon"])
        x2 = x2 @ W[f"h.{i}.mlp.c_fc.weight"] + W[f"h.{i}.mlp.c_fc.bias"]            
        x2 = GeLU(x2)
        x2 = x2 @ W[f"h.{i}.mlp.c_proj.weight"] + W[f"h.{i}.mlp.c_proj.bias"]
        x = x + x2 # res

    x = layer_norm(x, W[f"ln_f.weight"], W[f"ln_f.bias"], config["layer_norm_epsilon"])
    x = x @ W["wte.weight"].T

    logits = x[-1]
    logits = logits / temperature
    probs = softmax(logits)

    next_id = np.random.choice(len(probs), p=probs)
    return int(next_id)


def inference(chat: str, weights: np.array, config: dict, tokens: dict) -> None:
    print(chat, end="", flush=True)
    
    ids = tokens.encode(chat).ids
    init_len = len(ids)
    kv_cache = []

    while len(ids) < config["n_ctx"]:
        if MAX_LEN > 0 and len(ids) > MAX_LEN:
            break
        prefil = True if len(ids) == init_len else False
        next_token = pred_next_tk(ids, weights, config, kv_cache, prefil=prefil)
        if next_token == config["eos_token_id"]:
            break
        ids.append(next_token)
        text = tokens.decode([next_token])
        print(text, end="", flush=True)


def main():
    args = parse_args()

    config = parse_json(args.model_config)
    tokens = Tokenizer.from_file(args.tokenizer)
    weights = load_file(args.weights)
    
    inference(args.context, weights, config, tokens)


if __name__ == "__main__":
    main()
