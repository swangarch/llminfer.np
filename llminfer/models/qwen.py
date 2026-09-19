import math
import numpy as np
from .BaseModel import BaseModel
from .utils import rms_norm, rope, softmax, SiLU
from .ModelRegistry import ModelRegistry


@ModelRegistry.register
class Qwen(BaseModel):
    name: str = "qwen2.5"
    chat_supported: bool = True

    def __init__(self, weights: np.array, config: dict, tokens: dict):
        super().__init__(weights, config, tokens)
        self.kv_cache = []

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    def pred_next_token(self, ids: list, prefill: bool = False, kv_cache_enabled: bool = False,
                    temperature: float = 0.8) -> int:

        if not kv_cache_enabled or prefill:
            x = self.weights["model.embed_tokens.weight"][ids]
        else:
            x = self.weights["model.embed_tokens.weight"][[ids[-1]]]

        seqlen = x.shape[0]
        mask = self.np.triu(self.np.full((seqlen, seqlen), -self.np.inf), k = 1)

        for i in range(self.config["num_hidden_layers"]):
            residual = x
            h = rms_norm(x, self.weights[f"model.layers.{i}.input_layernorm.weight"], eps=self.config["rms_norm_eps"], np=self.np)

            Q_weights = self.weights[f"model.layers.{i}.self_attn.q_proj.weight"].T
            Q_biases = self.weights[f"model.layers.{i}.self_attn.q_proj.bias"]
            K_weights = self.weights[f"model.layers.{i}.self_attn.k_proj.weight"].T
            K_biases = self.weights[f"model.layers.{i}.self_attn.k_proj.bias"]
            V_weights = self.weights[f"model.layers.{i}.self_attn.v_proj.weight"].T
            V_biases = self.weights[f"model.layers.{i}.self_attn.v_proj.bias"]

            Q = h @ Q_weights + Q_biases
            if not kv_cache_enabled or prefill:
                K = h @ K_weights + K_biases
                V = h @ V_weights + V_biases
            else:
                K = self.np.concatenate([self.kv_cache[i]["K"], (h[-1:] @ K_weights + K_biases)], axis=0)
                V = self.np.concatenate([self.kv_cache[i]["V"], (h[-1:] @ V_weights + V_biases)], axis=0)

            Q_heads = self.np.stack(self.np.split(Q, self.config["num_attention_heads"], axis=1), axis=0)
            K_heads = self.np.stack(self.np.split(K, self.config["num_key_value_heads"], axis=1), axis=0)
            V_heads = self.np.stack(self.np.split(V, self.config["num_key_value_heads"], axis=1), axis=0)

            if kv_cache_enabled:
                if prefill:
                    self.kv_cache.append({"K": K, "V": V})
                else:   
                    self.kv_cache[i] = {"K": K, "V": V}

            attn_heads = []

            q_start_pos = K.shape[0] - Q.shape[0]
            Q_heads = rope(Q_heads, self.config["rope_theta"], q_start_pos, np=self.np)
            K_heads = rope(K_heads, self.config["rope_theta"], 0, np=self.np)
            repeat_time = self.config["num_attention_heads"] // self.config["num_key_value_heads"]
            K_heads = self.np.repeat(K_heads, repeat_time, axis=0)
            V_heads = self.np.repeat(V_heads, repeat_time, axis=0)

            for j in range(self.config["num_attention_heads"]):
                hidden_dim = self.config["hidden_size"] // self.config["num_attention_heads"]
                score = Q_heads[j] @ K_heads[j].T / math.sqrt(hidden_dim)
                score = score + mask
                attn_head = softmax(score, np=self.np) @ V_heads[j] 
                attn_heads.append(attn_head)

            multi_head = self.np.concatenate(attn_heads, axis=-1)
            attn_out = multi_head @ self.weights[f"model.layers.{i}.self_attn.o_proj.weight"].T
            x = residual + attn_out

            residual = x
            h = rms_norm(x, self.weights[f"model.layers.{i}.post_attention_layernorm.weight"], eps=self.config["rms_norm_eps"], np=self.np)
            gate = SiLU(h @ self.weights[f"model.layers.{i}.mlp.gate_proj.weight"].T, np=self.np)
            up = h @ self.weights[f"model.layers.{i}.mlp.up_proj.weight"].T
            mid = gate * up
            out = mid @ self.weights[f"model.layers.{i}.mlp.down_proj.weight"].T
            x = residual + out

        x = rms_norm(x, self.weights["model.norm.weight"], self.config["rms_norm_eps"], np=self.np)
        x = x @ self.weights["model.embed_tokens.weight"].T

        logits = x[-1]
        logits = logits / temperature
        probs = softmax(logits, np=self.np)

        next_id = self.np.random.choice(len(probs), size=1, p=probs)[0]
        return int(next_id)

    def inference_no_chat(self, context: str, kv_cache_enabled: bool = False, max_len: int = 350) -> None:
        print(context, end="", flush=True)
        
        ids = self.tokens.encode(context).ids
        init_len = len(ids)

        while len(ids) < max_len:
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

    def inference_chat(self, context: str, kv_cache_enabled: bool = False, max_len: int = 350) -> None:

        context = f"<|im_start|>system\n{context}<|im_end|>"
        print("Assitant: Hi, i am an AI assistant.", flush=True)

        user_input = input("User: ")
        context += f"<|im_start|>user\n{user_input}<|im_end|><|im_start|>assitant\n"
        print("Assistant: ", end="")
        chat_buffer = ""
        
        ids = self.tokens.encode(context).ids
        init_len = len(ids)

        while len(ids) < max_len:
            if max_len > 0 and len(ids) > max_len:
                break
            prefill = True if len(ids) == init_len else False
            next_token = self.pred_next_token(ids, prefill=prefill, 
                                              kv_cache_enabled=kv_cache_enabled)

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
                user_input_ids = self.tokens.encode(f"{user_input_text}<|im_start|>assistant\n").ids
                ids += user_input_ids
                print("Assistant: ", end="")
                continue

            elif next_token == self.config["eos_token_id"]:
                #handle last chat
                break

            text = self.tokens.decode([next_token])
            chat_buffer += text
      
            print(text, end="", flush=True)
            ids.append(next_token)


    def inference(self, context: str, kv_cache_enabled: bool = False, max_len: int = 350, chat: bool = False) -> None:
        self.kv_cache.clear()
        if chat:
            self.inference_chat(context, kv_cache_enabled=kv_cache_enabled, max_len=max_len)
        else:
            self.inference_no_chat(context, kv_cache_enabled=kv_cache_enabled, max_len=max_len)