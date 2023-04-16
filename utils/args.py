import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description="Define inference parameters.")

    parser.add_argument("--style_weight", type=int, default=1000000000)
    parser.add_argument("--content_weight", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--style", type=str, default="picasso")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.device = "cuda"
    if torch.backends.mps.is_available():
        args.device = "mps"

    return args
