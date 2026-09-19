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


def main():
    args = parse_args()
    if not check_model_downloaded(args.model):
        return
    config = parse_json(f"model/{args.model}/config.json")
    tokens = Tokenizer.from_file(f"model/{args.model}/tokenizer.json")
    weights = load_file(f"model/{args.model}/model.safetensors")

    try:
        model = ModelRegistry.get_model(args.model)(weights, config, tokens)
        model.use_cuda(args.cuda)
        model.print_model()
        model.set_seed(args.seed)
        model.inference(args.context, args.kv_cache, args.max_len, args.chat)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\nError: {e}.")


if __name__ == "__main__":
    main()
