"""Diagnose whether CLIP-Count distinguishes different text prompts.

This script is read-only with respect to the model: it loads a checkpoint,
runs several prompts on the same image, and compares token IDs, text
embeddings, raw density maps, and predicted counts.
"""

import argparse
from itertools import combinations
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from run import Model
from util import misc
from util.constant import SCALE_FACTOR


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Compare CLIP-Count outputs for multiple prompts on one image."
    )
    parser.add_argument("--ckpt", type=Path, default="lightning_logs/exp0828/version_1/checkpoints/epoch=179-val_mae=15.21.ckpt", help="Lightning .ckpt file")
    parser.add_argument("--image", type=Path, default="data/FSC/images_384_VarV2/41.jpg", help="Input image")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["car", "person", "apple"],
        help="Prompts to compare (default: car person apple)",
    )
    parser.add_argument("--stride", default=128, type=int)
    return parser


def load_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    image = TF.pil_to_tensor(image).unsqueeze(0).to(device=device, dtype=torch.float32)
    image = image / 255.0

    original_height, original_width = image.shape[-2:]
    resized_width = max(384, round(original_width * 384 / original_height))
    image = TF.resize(image, [384, resized_width], antialias=True)
    return torch.clamp(image, 0, 1), resized_width


def get_text_embeddings(model, prompts, device):
    tokens = clip.tokenize(prompts).to(device)
    # CLIP keeps its pretrained weights in float16. Match the regular model
    # forward path, which also runs the text encoder under CUDA autocast.
    with torch.amp.autocast("cuda"):
        embeddings = model.model.text_encoder(tokens)
    embeddings = embeddings.float().squeeze(1)
    return tokens, embeddings


def predict_density(model, image, prompt, stride, raw_width):
    patches, _ = misc.sliding_window(image, stride=stride)
    patches = torch.from_numpy(patches).to(device=image.device, dtype=torch.float32)
    patch_prompts = [prompt] * patches.shape[0]

    with torch.amp.autocast("cuda"):
        output = model(patches, patch_prompts)

    output = misc.window_composite(output.unsqueeze(1), stride=stride).squeeze(1)
    return output[:, :, :raw_width]


def print_tokens(prompts, tokens):
    print("\n=== Tokenization ===")
    for prompt, token_row in zip(prompts, tokens):
        eot_index = int(token_row.argmax().item())
        active_tokens = token_row[: eot_index + 1].tolist()
        print(f"{prompt!r}: eot_index={eot_index}, tokens={active_tokens}")


def print_pairwise_results(prompts, embeddings, densities, counts):
    print("\n=== Pairwise prompt comparison ===")
    for first, second in combinations(range(len(prompts)), 2):
        cosine = F.cosine_similarity(
            embeddings[first].unsqueeze(0), embeddings[second].unsqueeze(0)
        ).item()
        embedding_max_diff = (
            embeddings[first] - embeddings[second]
        ).abs().max().item()
        density_mean_diff = (
            densities[first] - densities[second]
        ).abs().mean().item()
        count_diff = abs(counts[first] - counts[second])
        embeddings_equal = torch.allclose(
            embeddings[first], embeddings[second], rtol=1e-5, atol=1e-6
        )

        print(f"\n{prompts[first]!r} vs {prompts[second]!r}")
        print(f"  embeddings_allclose : {embeddings_equal}")
        print(f"  embedding_cosine    : {cosine:.8f}")
        print(f"  embedding_max_diff  : {embedding_max_diff:.8e}")
        print(f"  density_mean_diff   : {density_mean_diff:.8e}")
        print(f"  count_difference    : {count_diff:.8f}")


def main():
    args = get_args_parser().parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This project requires a CUDA GPU for CLIP-Count inference.")
    if not args.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if len(args.prompts) < 2:
        raise ValueError("Provide at least two prompts for comparison.")

    device = torch.device("cuda")
    model = Model.load_from_checkpoint(str(args.ckpt), strict=False)
    model.model = model.model.to(device)
    model.eval()

    image, raw_width = load_image(args.image, device)

    with torch.inference_mode():
        tokens, embeddings = get_text_embeddings(model, args.prompts, device)

        densities = []
        counts = []
        for prompt in args.prompts:
            density = predict_density(
                model.model, image, prompt, args.stride, raw_width
            )
            count = torch.sum(density[0] / SCALE_FACTOR).item()
            densities.append(density[0].float())
            counts.append(count)

    print_tokens(args.prompts, tokens.cpu())

    print("\n=== Raw model outputs (before visualization normalization) ===")
    for prompt, density, count in zip(args.prompts, densities, counts):
        print(
            f"{prompt!r}: count={count:.8f}, "
            f"density_min={density.min().item():.8e}, "
            f"density_max={density.max().item():.8e}, "
            f"density_mean={density.mean().item():.8e}"
        )

    print_pairwise_results(args.prompts, embeddings, densities, counts)


if __name__ == "__main__":
    main()
