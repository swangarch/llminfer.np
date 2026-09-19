from pathlib import Path
from safetensors.torch import load_file, save_file


def main():
    src = Path("model/qwen2.5/model.safetensors")
    dst = Path("model/qwen2.5/model.safetensors")

    print("Start to convert weight to fp32.")

    weights = load_file(src, device="cpu")
    weights_fp32 = {
        name: tensor.float().contiguous()
        for name, tensor in weights.items()
    }
    save_file(weights_fp32, str(dst))
    print("Weights has been converted to fp32.")


if __name__ == "__main__":
    main()