"""Diagnose whether CLIP-Count distinguishes different text prompts.

This script is read-only with respect to the model: it loads a checkpoint,
runs several prompts on the same image, and compares token IDs, text
embeddings, patch-text similarity maps, raw density maps, and predicted counts.

The similarity branch is diagnostic only: it does not alter the features sent
to FIM or the density decoder.
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
    parser.add_argument("--ckpt", type=Path, default="lightning_logs/similarity-gate/version_0/checkpoints/epoch=12-val_mae=14.34.ckpt", help="Lightning .ckpt file")
    parser.add_argument("--image", type=Path, default="data/FSC/images_384_VarV2/3.jpg", help="Input image")
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["hot air balloons", "tomatoes", "car", "person", "apple"],
        help="Prompts to compare (default: car person apple)",
    )
    parser.add_argument("--stride", default=128, type=int)
    parser.add_argument(
        "--gate-temperature",
        default=0.1,
        type=float,
        help="Sigmoid temperature used by the model similarity gate",
    )
    parser.add_argument(
        "--gate-threshold",
        default=0.0,
        type=float,
        help="Cosine-similarity threshold used by the model gate",
    )
    parser.add_argument(
        "--gate-residual",
        default=0.2,
        type=float,
        help="Minimum decoder-feature scale when the model gate is enabled",
    )
    parser.add_argument(
        "--use-similarity-gate",
        default=True,
        type=misc.str2bool,
        help="Apply the similarity gate to the model prediction path",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("diagnose_prompt_output"),
        type=Path,
        help="Directory for similarity and gate visualizations",
    )
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


def predict_density_and_similarity(model, image, prompt, stride, raw_width):
    patches, _ = misc.sliding_window(image, stride=stride)
    patches = torch.from_numpy(patches).to(device=image.device, dtype=torch.float32)
    patch_prompts = [prompt] * patches.shape[0]

    with torch.amp.autocast("cuda"):
        output, extra_out = model(patches, patch_prompts, return_extra=True)

    output = misc.window_composite(output.unsqueeze(1), stride=stride).squeeze(1)
    density = output[:, :, :raw_width]

    patch_count = extra_out["patch_text_similarity"].shape[1]
    grid_size = int(patch_count**0.5)
    if grid_size * grid_size != patch_count:
        raise ValueError(f"Patch count is not a square grid: {patch_count}")
    similarity = extra_out["patch_text_similarity"].reshape(
        -1, 1, grid_size, grid_size
    )
    similarity = F.interpolate(
        similarity.float(), size=(384, 384), mode="bilinear", align_corners=False
    )
    similarity = misc.window_composite(similarity, stride=stride)
    similarity = similarity[:, :, :raw_width]
    gate = extra_out["patch_text_gate"].reshape(-1, 1, grid_size, grid_size)
    gate = F.interpolate(
        gate.float(), size=(384, 384), mode="bilinear", align_corners=False
    )
    gate = misc.window_composite(gate, stride=stride)
    gate = gate[:, :, :raw_width]
    return density, similarity, gate


def safe_filename(prompt):
    name = "".join(character if character.isalnum() else "_" for character in prompt)
    return name.strip("_") or "prompt"


def save_map(tensor, path, value_range):
    values = tensor.detach().float().cpu()
    minimum, maximum = value_range
    normalized = ((values - minimum) / (maximum - minimum)).clamp(0, 1)
    image = Image.fromarray((normalized.numpy() * 255).astype(np.uint8), mode="L")
    image.save(path)


def print_tokens(prompts, tokens):
    print("\n=== Tokenization ===")
    for prompt, token_row in zip(prompts, tokens):
        eot_index = int(token_row.argmax().item())
        active_tokens = token_row[: eot_index + 1].tolist()
        print(f"{prompt!r}: eot_index={eot_index}, tokens={active_tokens}")


def print_pairwise_results(prompts, embeddings, densities, similarities, gates, counts):
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
        similarity_mean_diff = (
            similarities[first] - similarities[second]
        ).abs().mean().item()
        gate_mean_diff = (gates[first] - gates[second]).abs().mean().item()
        count_diff = abs(counts[first] - counts[second])
        embeddings_equal = torch.allclose(
            embeddings[first], embeddings[second], rtol=1e-5, atol=1e-6
        )

        print(f"\n{prompts[first]!r} vs {prompts[second]!r}")
        print(f"  embeddings_allclose : {embeddings_equal}")
        print(f"  embedding_cosine    : {cosine:.8f}")
        print(f"  embedding_max_diff  : {embedding_max_diff:.8e}")
        print(f"  similarity_mean_diff: {similarity_mean_diff:.8e}")
        print(f"  gate_mean_diff      : {gate_mean_diff:.8e}")
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
    if args.gate_temperature <= 0:
        raise ValueError("--gate-temperature must be greater than zero.")
    if not 0 <= args.gate_residual <= 1:
        raise ValueError("--gate-residual must be between zero and one.")

    device = torch.device("cuda")
    model = Model.load_from_checkpoint(str(args.ckpt), strict=False)
    model.model = model.model.to(device)
    model.model.use_similarity_gate = args.use_similarity_gate
    model.model.gate_temperature = args.gate_temperature
    model.model.gate_threshold = args.gate_threshold
    model.model.gate_residual = args.gate_residual
    model.eval()

    image, raw_width = load_image(args.image, device)

    with torch.inference_mode():
        tokens, embeddings = get_text_embeddings(model, args.prompts, device)

        densities = []
        similarities = []
        gates = []
        counts = []
        for prompt in args.prompts:
            density, similarity, gate = predict_density_and_similarity(
                model.model,
                image,
                prompt,
                args.stride,
                raw_width,
            )
            count = torch.sum(density[0] / SCALE_FACTOR).item()
            densities.append(density[0].float())
            similarities.append(similarity[0].float())
            gates.append(gate[0].float())
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

    gate_status = "applied to prediction" if args.use_similarity_gate else "diagnostic only"
    print(f"\n=== Patch-text similarity ({gate_status}) ===")
    for prompt, similarity, gate in zip(args.prompts, similarities, gates):
        print(
            f"{prompt!r}: "
            f"similarity_min={similarity.min().item():.8f}, "
            f"similarity_max={similarity.max().item():.8f}, "
            f"similarity_mean={similarity.mean().item():.8f}, "
            f"similarity_std={similarity.std().item():.8f}, "
            f"gate_min={gate.min().item():.8f}, "
            f"gate_max={gate.max().item():.8f}, "
            f"gate_mean={gate.mean().item():.8f}, "
            f"gate_std={gate.std().item():.8f}"
        )

    print_pairwise_results(
        args.prompts, embeddings, densities, similarities, gates, counts
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for prompt, similarity, gate in zip(args.prompts, similarities, gates):
        name = safe_filename(prompt)
        # Use fixed ranges so brightness remains comparable across prompts.
        save_map(
            similarity,
            args.output_dir / f"{name}_similarity.png",
            value_range=(-1.0, 1.0),
        )
        save_map(
            gate,
            args.output_dir / f"{name}_gate.png",
            value_range=(0.0, 1.0),
        )
    print(f"\nSaved diagnostic maps to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
