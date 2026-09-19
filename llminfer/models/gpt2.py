import numpy as np
import math
from .BaseModel import BaseModel
from .utils import layer_norm, softmax, GeLU
from .ModelRegistry import ModelRegistry


@ModelRegistry.register
class GPT2(BaseModel):
    name: str = "gpt2"
    chat_supported: bool = False

    def __init__(self, weights: np.array, config: dict, tokens: dict):
        super().__init__(weights, config, tokens)
        self.kv_cache = []

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    def pred_next_token(self, ids: list, prefill: bool = False, kv_cache_enabled: bool = False,
                    temperature: float = 0.8) -> int:
        if not kv_cache_enabled or prefill:
            x = self.weights["wte.weight"][ids] + self.weights["wpe.weight"][self.np.arange(len(ids))]
        else:
            x = self.weights["wte.weight"][[ids[-1]]] + self.weights["wpe.weight"][[len(ids) - 1]]
        seqlen = x.shape[0]
        mask = self.np.triu(self.np.full((seqlen, seqlen), -self.np.inf), k=1)

        for i in range(self.config["n_layer"]):
            x1 = layer_norm(x, self.weights[f"h.{i}.ln_1.weight"], self.weights[f"h.{i}.ln_1.bias"], self.config["layer_norm_epsilon"], np=self.np)
            if not kv_cache_enabled or prefill:
                QKV = x1 @ self.weights[f"h.{i}.attn.c_attn.weight"] + self.weights[f"h.{i}.attn.c_attn.bias"]
                Q, K, V = self.np.split(QKV, 3, axis=-1)
            else:
                Wq, Wk, Wv = self.np.split(self.weights[f"h.{i}.attn.c_attn.weight"], 3, axis=-1)
                Bq, Bk, Bv = self.np.split(self.weights[f"h.{i}.attn.c_attn.bias"], 3, axis=-1)
                Q = x1 @ Wq + Bq
                K = self.np.concatenate([self.kv_cache[i]["K"], (x1[-1, :] @ Wk + Bk).reshape(1, -1)], axis=0)
                V = self.np.concatenate([self.kv_cache[i]["V"], (x1[-1, :] @ Wv + Bv).reshape(1, -1)], axis=0)

            if prefill:
                self.kv_cache.append({"K": K, "V": V})
            else:   
                self.kv_cache[i] = {"K": K, "V": V}
            Q_heads = self.np.split(Q, self.config["n_head"], axis=-1)
            K_heads = self.np.split(K, self.config["n_head"], axis=-1)
            V_heads = self.np.split(V, self.config["n_head"], axis=-1)

            attn_heads = []
            for j in range(self.config["n_head"]):
                score = Q_heads[j] @ K_heads[j].T / (math.sqrt(K_heads[j].shape[1]))
                if not kv_cache_enabled or prefill:
                    score = score + mask
                attn_head = softmax(score, np=self.np) @ V_heads[j]
                attn_heads.append(attn_head)
            
            multi_head = self.np.concatenate(attn_heads, axis=-1)
            x1 = multi_head @ self.weights[f"h.{i}.attn.c_proj.weight"] + self.weights[f"h.{i}.attn.c_proj.bias"]
            x = x + x1 # res

            x2 = layer_norm(x, self.weights[f"h.{i}.ln_2.weight"], self.weights[f"h.{i}.ln_2.bias"], self.config["layer_norm_epsilon"], np=self.np)
            x2 = x2 @ self.weights[f"h.{i}.mlp.c_fc.weight"] + self.weights[f"h.{i}.mlp.c_fc.bias"]            
            x2 = GeLU(x2, np=self.np)
            x2 = x2 @ self.weights[f"h.{i}.mlp.c_proj.weight"] + self.weights[f"h.{i}.mlp.c_proj.bias"]
            x = x + x2 # res

        x = layer_norm(x, self.weights[f"ln_f.weight"], self.weights[f"ln_f.bias"], self.config["layer_norm_epsilon"], np=self.np)
        x = x @ self.weights["wte.weight"].T

        logits = x[-1]
        logits = logits / temperature
        probs = softmax(logits, np=self.np)

        next_id = self.np.random.choice(len(probs), size=1, p=probs)[0]
        return int(next_id)

    def inference(self, context: str, kv_cache_enabled: bool = False, max_len: int = 150, chat: bool = False) -> None:
        self.kv_cache.clear()
        if chat:
            print("Chat is not supported for GPT2")
            print()

        print(context, end="", flush=True)
        
        ids = self.tokens.encode(context).ids
        init_len = len(ids)
        out_text = context

        while len(ids) < self.config["n_ctx"]:
            if max_len > 0 and len(ids) > max_len:
                break
            prefill = True if len(ids) == init_len else False
            next_token = self.pred_next_token(ids, prefill=prefill,
                                    kv_cache_enabled=kv_cache_enabled)
            if next_token == self.config["eos_token_id"]:
                break
            ids.append(next_token)
            text = self.tokens.decode([next_token])
            print(text, end="", flush=True)
            out_text += text
        return out_text
