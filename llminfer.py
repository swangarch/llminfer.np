import numpy as np
import os
from safetensors.numpy import load_file
from tokenizers import Tokenizer
import argparse
from llminfer import ModelRegistry, parse_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-cu", "--cuda", action="store_true")
    parser.add_argument("-kv", "--kv_cache", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=422)
    parser.add_argument("-ml", "--max_len", type=int, default=150)
    parser.add_argument("-m", "--model", type=str, default=f"qwen2.5")
    parser.add_argument("-ch", "--chat", default=False, action="store_true")
    parser.add_argument("-c", "--context", type=str, default="""The capital of France is Paris. Paris is a beautiful historical city.""")

    args = parser.parse_args()
    return args


def check_model_downloaded(model: str) -> bool:
    if not os.path.exists(f"model/{model}"):
        print(f"Model {model} not exist, download model first.")
        return False
    return True


def use_cuda(weights: dict, cuda: bool = False) -> str:
    if cuda == True:
        global np
        try:
            import cupy as np
            for k, v in weights.items():
                weights[k] = np.array(v)
            print("Using GPU CUDA accelaration.")
            return "gpu"
        except ImportError:
            print("Failed to import Cupy, using Numpy on CPU.")
    else:
        print("Using CPU.")
    return "cpu"


def main():
    args = parse_args()

    if not check_model_downloaded(args.model):
        return

    config_path = f"model/{args.model}/config.json"
    tokenizer_path = f"model/{args.model}/tokenizer.json"
    weights_path = f"model/{args.model}/model.safetensors"

    config = parse_json(config_path)
    tokens = Tokenizer.from_file(tokenizer_path)
    weights = load_file(weights_path)

    print("=" * 40)
    print(f"Model -- {args.model}")
    device = use_cuda(weights, args.cuda)
    print("KV cache enabled." if args.kv_cache else "KV cache disabled.")
    print("=" * 40)
    print()
    np.random.seed(args.seed)
    
    model = ModelRegistry.get_model(args.model)(weights, config, tokens)

    model.inference(args.context, args.kv_cache, args.max_len, args.chat)


if __name__ == "__main__":
    main()
