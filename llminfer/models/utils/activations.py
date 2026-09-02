import numpy as np
import math


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
