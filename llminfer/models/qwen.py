import numpy as np
import math
from .BaseModel import BaseModel
from .utils import rms_norm, rope, softmax, SiLU
from .ModelRegistry import ModelRegistry


@ModelRegistry.register
class Qwen(BaseModel):
    name: str = "qwen2.5"
    chat_supported: bool = True

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @staticmethod
    def pred_next_token(ids: list, W: np.array, config: dict, kv_cache: list = [],
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
    def inference_no_chat(context: str, weights: np.array, config: dict, tokens: dict,
                kv_cache_enabled: bool = False, max_len: int = 350) -> None:
        print(context, end="", flush=True)
        
        ids = tokens.encode(context).ids

        while len(ids) < max_len:
            if max_len > 0 and len(ids) > max_len:
                break
            next_token = Qwen.pred_next_token(ids, weights, config)
            if next_token == config["eos_token_id"]:
                break
            ids.append(next_token)
            text = tokens.decode([next_token])
            print(text, end="", flush=True)

    @staticmethod
    def inference_chat(context: str, weights: np.array, config: dict, tokens: dict,
                kv_cache_enabled: bool = False, max_len: int = 350) -> None:

        context = f"<|im_start|>system\n{context}<|im_end|>"
        print("Assitant: Hi, i am an AI assistant.", flush=True)

        user_input = input("User: ")
        context += f"<|im_start|>user\n{user_input}<|im_end|><|im_start|>assitant\n"
        print("Assistant: ", end="")
        chat_buffer = ""
        
        ids = tokens.encode(context).ids

        while len(ids) < max_len:
            if max_len > 0 and len(ids) > max_len:
                break
            next_token = Qwen.pred_next_token(ids, weights, config)

            if next_token == 151644: #<|im_start|>
                ids.append(next_token)
                chat_buffer = ""
                print(f"Assistant: ", end="")
                continue

            elif next_token == 151645: #<|im_end|>
                ids.append(next_token)
                print()

                user_input = input("User: ")
                user_input_text = f"<|im_start|>user\n{user_input}<|im_end|>"
                user_input_ids = tokens.encode(f"{user_input_text}<|im_start|>assistant\n").ids
                ids += user_input_ids

                print("Assistant: ", end="")
                continue

            elif next_token == config["eos_token_id"]:
                #handle last chat
                break

            text = tokens.decode([next_token])
            chat_buffer += text
      
            print(text, end="", flush=True)
            ids.append(next_token)


    def inference(context: str, weights: np.array, config: dict, tokens: dict,
                kv_cache_enabled: bool = False, max_len: int = 350, chat: bool = False) -> None:
        if chat:
            Qwen.inference_chat(context, weights, config, tokens, 
                                   kv_cache_enabled=kv_cache_enabled, 
                                   max_len=max_len)
        else:
            Qwen.inference_no_chat(context, weights, config, tokens, 
                                   kv_cache_enabled=kv_cache_enabled, 
                                   max_len=max_len)