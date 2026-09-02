import numpy as np


def layer_norm(x: np.array, weight: np.array, bias: np.array, eps: float = 1e-5) -> np.array:
    mu   = x.mean(axis=-1, keepdims=True) # (seq, 1）
    var  = x.var(axis=-1, keepdims=True)  # (seq, 1)
    norm = (x - mu) / np.sqrt(var + eps)  # (seq, 768)
    return norm * weight + bias


def rms_norm(x: np.array, weight: np.array, eps: float = 1e-5) -> np.array:
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * weight
