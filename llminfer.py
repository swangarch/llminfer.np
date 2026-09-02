import numpy as np
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import argparse
from llminfer import infer_qwen, parse_json


MAX_LEN = 150
np.random.seed(422)


def parse_args():
    parser = argparse.ArgumentParser()
    # Add a option to show model structure

    model = "qwen2.5"

    parser.add_argument("-mc", "--model_config", type=str, default=f"model/{model}/config.json")
    parser.add_argument("-t", "--tokenizer", type=str, default=f"model/{model}/tokenizer.json")
    parser.add_argument("-w", "--weights", type=str, default=f"model/{model}/model_fp32.safetensors")
    parser.add_argument("-c", "--context", type=str, default="""The following is a conversation between a User and a helpful Assistant.
       
    User: What is the capital of France?
    Assistant: The capital of France is Paris.

    User: Tell me more about France.
    Assistant:""")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    config = parse_json(args.model_config)
    tokens = Tokenizer.from_file(args.tokenizer)
    weights = load_file(args.weights)
    
    infer_qwen(args.context, weights, config, tokens)


if __name__ == "__main__":
    main()
