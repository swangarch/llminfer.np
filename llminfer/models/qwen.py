import numpy as np
import math
from .BaseModel import BaseModel
from .utils import rms_norm, rope, softmax, SiLU
from .ModelRegistry import ModelRegistry


@ModelRegistry.register
class Qwen(BaseModel):
    name: str = "qwen2.5"

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @staticmethod
    def pred_next_tk(ids: list, W: np.array, config: dict, kv_cache: list = [],
                    prefill: bool = False, kv_cache_enabled: bool = False,
                    temperature: float = 0.8) -> int:
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

    @staticmethod
    def inference(context: str, weights: np.array, config: dict, tokens: dict,
                kv_cache_enabled: bool = False, max_len: int = 150) -> None:
        print(context, end="", flush=True)
        
        ids = tokens.encode(context).ids

        while len(ids) < max_len:
            if max_len > 0 and len(ids) > max_len:
                break
            next_token = Qwen.pred_next_tk(ids, weights, config)
            if next_token == config["eos_token_id"]:
                break
            ids.append(next_token)
            text = tokens.decode([next_token])
            print(text, end="", flush=True)
