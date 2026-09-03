
import numpy as np
import math
from .BaseModel import BaseModel
from .utils import layer_norm, softmax, GeLU
from .ModelRegistry import ModelRegistry


@ModelRegistry.register
class GPT2(BaseModel):
    name: str = "gpt2"

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @staticmethod
    def pred_next_tk(ids: list, W: np.array, config: dict, kv_cache: list = [],
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
                K = np.concatenate([kv_cache[i]["K"], (x1[-1, :] @ Wk + Bk).reshape(1, -1)], axis=0)
                V = np.concatenate([kv_cache[i]["V"], (x1[-1, :] @ Wv + Bv).reshape(1, -1)], axis=0)

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

    @staticmethod
    def inference(context: str, weights: np.array, config: dict, tokens: dict,
                kv_cache_enabled: bool = False, max_len: int = 150) -> None:
        print(context, end="", flush=True)
        
        ids = tokens.encode(context).ids
        init_len = len(ids)
        kv_cache = []
        out_text = context

        while len(ids) < config["n_ctx"]:
            if max_len > 0 and len(ids) > max_len:
                break
            prefill = True if len(ids) == init_len else False
            next_token = GPT2.pred_next_tk(ids, weights, config, kv_cache, prefill=prefill,
                                    kv_cache_enabled=kv_cache_enabled)
            if next_token == config["eos_token_id"]:
                break
            ids.append(next_token)
            text = tokens.decode([next_token])
            print(text, end="", flush=True)
            out_text += text
        return out_text
