class ModelRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, model: type) -> type:

        name = model.get_name()
        if name in cls._registry:
            raise ValueError("Model already in the registry.")

        cls._registry[name] = model
        return model

    @classmethod
    def get_model(cls, name: str) -> type:
        if name not in cls._registry:
            raise ValueError("Model not in the registry.")
        return cls._registry[name]
