import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import argparse
from llminfer import ModelRegistry, parse_json


MAX_LEN = 150
np.random.seed(422)


def parse_args():
    parser = argparse.ArgumentParser()
    # Add a option to show model structure

    parser.add_argument("-m", "--model", type=str, default=f"qwen2.5")
    parser.add_argument("-c", "--context", type=str, default="""The following is a conversation between a User and a helpful Assistant.
       
    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant:""")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config_path = f"model/{args.model}/config.json"
    tokenizer_path = f"model/{args.model}/tokenizer.json"
    weights_path = f"model/{args.model}/model.safetensors"

    config = parse_json(config_path)
    tokens = Tokenizer.from_file(tokenizer_path)
    weights = load_file(weights_path)
    
    ModelRegistry.get_model(args.model).inference(args.context, weights, config, tokens)


if __name__ == "__main__":
    main()
