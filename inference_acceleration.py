import json
from safetensors.numpy import load_file
import numpy as np
from tokenizers import Tokenizer
import math
import argparse


MAX_LEN = 150
RD_SEED = 422


def parse_args(argv = None):
    parser = argparse.ArgumentParser()

    parser.add_argument("-mc", "--model_config", type=str, default="model/config.json")
    parser.add_argument("-t", "--tokenizer", type=str, default="model/tokenizer.json")
    parser.add_argument("-w", "--weights", type=str, default="model/model.safetensors")
    parser.add_argument("-cu", "--cuda", action="store_true")
    parser.add_argument("-kv", "--kv_cache", action="store_true")
    parser.add_argument("-c", "--context", type=str, default="""The following is a conversation between a User and a helpful Assistant.

    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant:""")

    args = parser.parse_args(argv)
    return args


def parse_json(path: str) -> dict:
    try:
        with open(path, mode="r", encoding="UTF-8") as f:
            content = json.load(f)
            return content
    except Exception as e:
        print("\033[31mError:", e, "\033[0m")
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
                 prefill: bool = False, kv_cache_enabled: bool = False,
                 temperature: float = 0.8) -> int:
    if not kv_cache_enabled or prefill:
        x = W["wte.weight"][ids] + W["wpe.weight"][np.arange(len(ids))]
    else:
        pos = len(ids) - 1
        x = W["wte.weight"][[ids[-1]]] + W["wpe.weight"][[pos]]
    seqlen = x.shape[0]
    mask = np.triu(np.full((seqlen, seqlen), -np.inf), k=1)

    for i in range(config["n_layer"]):
        x1 = layer_norm(x, W[f"h.{i}.ln_1.weight"], W[f"h.{i}.ln_1.bias"], config["layer_norm_epsilon"])
        if not kv_cache_enabled or prefill:
            QKV = x1 @ W[f"h.{i}.attn.c_attn.weight"] + W[f"h.{i}.attn.c_attn.bias"]
            Q, K, V = np.split(QKV, 3, axis=-1)
        else:
            Wq, Wk, Wv = np.split(W[f"h.{i}.attn.c_attn.weight"], 3, axis=-1)
            Bq, Bk, Bv = np.split(W[f"h.{i}.attn.c_attn.bias"], 3, axis=-1)
            Q = x1 @ Wq + Bq
            K = np.concatenate([kv_cache[i]["K"],  (x1[-1, :] @ Wk + Bk).reshape(1, -1)], axis=0)
            V = np.concatenate([kv_cache[i]["V"],  (x1[-1, :] @ Wv + Bv).reshape(1, -1)], axis=0)

        if not kv_cache_enabled or prefill:
            kv_cache.append({"K": K, "V": V})
        else:   
            kv_cache[i] = {"K": K, "V": V}
        Q_heads = np.split(Q, config["n_head"], axis=-1)
        K_heads = np.split(K, config["n_head"], axis=-1)
        V_heads = np.split(V, config["n_head"], axis=-1)

        attn_heads = []
        for j in range(config["n_head"]):
            score = Q_heads[j] @ K_heads[j].T / (math.sqrt(K_heads[j].shape[1]))
            if not kv_cache_enabled or prefill:
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

    next_id = np.random.choice(len(probs), size=1, p=probs)[0]
    return int(next_id)


def inference(chat: str, weights: np.array, config: dict, tokens: dict,
              kv_cache_enabled: bool = False) -> None:
    print(chat, end="", flush=True)
    
    ids = tokens.encode(chat).ids
    init_len = len(ids)
    kv_cache = []
    out_text = chat

    while len(ids) < config["n_ctx"]:
        if MAX_LEN > 0 and len(ids) > MAX_LEN:
            break
        prefill = True if len(ids) == init_len else False
        next_token = pred_next_tk(ids, weights, config, kv_cache, prefill=prefill,
                                  kv_cache_enabled=kv_cache_enabled)
        if next_token == config["eos_token_id"]:
            break
        ids.append(next_token)
        text = tokens.decode([next_token])
        print(text, end="", flush=True)
        out_text += text
    return out_text


def use_cuda(weights: dict, cuda: bool = False):
    if cuda == True:
        global np
        try:
            import cupy as np
            for k, v in weights.items():
                weights[k] = np.array(v)
            print("\033[32mUsing GPU CUDA accelaration.\033[0m")
        except ImportError:
            print("\033[32mFailed to import Cupy, using Numpy on CPU.\033[0m")
    else:
        print("\033[32mUsing CPU.\033[0m")


def main(argv = None):
    args = parse_args(argv)

    config = parse_json(args.model_config)
    tokens = Tokenizer.from_file(args.tokenizer)
    weights = load_file(args.weights)

    use_cuda(weights, args.cuda)
    np.random.seed(RD_SEED)
    
    return inference(args.context, weights, config, tokens, args.kv_cache)


if __name__ == "__main__":
    main()
