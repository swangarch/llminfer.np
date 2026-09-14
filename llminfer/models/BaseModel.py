
import numpy as np
from abc import ABC, abstractmethod


class BaseModel(ABC):
    name: str

    @abstractmethod
    def __init__(self, weights: np.array, config: dict, tokens: dict):
        raise NotImplementedError

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
