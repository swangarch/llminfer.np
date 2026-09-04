import json


def parse_json(path: str) -> dict:
    with open(path, mode="r", encoding="UTF-8") as f:
        return json.load(f)
