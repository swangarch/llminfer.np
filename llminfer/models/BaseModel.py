import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str

    def __init__(self, weights: np.array, config: dict, tokens: dict):
        self.weights = weights
        self.config = config
        self.tokens = tokens
        self.np = np
        self.device = "cpu"

    def print_model(self):
        print("=" * 40)
        print(f"= Model -- {self.name}")
        print(f"= Device -- {self.device}")
        print("=" * 40)
        print()

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @abstractmethod
    def pred_next_token(self, ids: list, prefill: bool = False, kv_cache_enabled: bool = False,
                    temperature: float = 0.8) -> int:
        raise NotImplementedError

    @abstractmethod
    def inference(self, context: str, kv_cache_enabled: bool = False, max_len: int = 350, chat: bool = False) -> None:
        raise NotImplementedError

    def use_cuda(self, cuda: bool = False) -> None:
        if cuda == True:
            try:
                import cupy as cp
                for k, v in self.weights.items():
                    self.weights[k] = cp.array(v)
                print("Using GPU CUDA accelaration.")
                self.np = cp
                self.device = "gpu"
            except ImportError as e:
                self.np = np
                self.device = "cpu"
                print("Failed to import Cupy, using Numpy on CPU.", e)
        else:
            print("Using CPU.")
            self.device = "cpu"

    def set_seed(self, seed: int) -> None:
        self.np.random.seed(seed)
