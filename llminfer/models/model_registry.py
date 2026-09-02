class models:
    _registry = {}


    def register(name: str, func: callable):
        models._registry[name] = func


    def get_model(name) -> callable:
        return models._registry[name]
