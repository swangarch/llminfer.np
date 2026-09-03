
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str

    @classmethod
    def get_name(cls) -> str:
        return cls.name

    @staticmethod
    @abstractmethod
    def pred_next_tk(ids: list, W: np.array, config: dict, kv_cache: list = [],
                    prefill: bool = False, kv_cache_enabled: bool = False,
                    temperature: float = 0.8) -> int:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def inference(context: str, weights: np.array, config: dict, tokens: dict,
                kv_cache_enabled: bool = False, max_len: int = 150) -> None:
        raise NotImplementedError
