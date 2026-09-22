"""Evaluate the released SBD-INR checkpoint on kodim23.png."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional

from model import BitPlaneSiren


ROOT = Path(__file__).resolve().parent.parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("coordinate_order") not in {"xy", "yx"}:
        raise ValueError("coordinate_order must be either 'xy' or 'yx'.")
    if config.get("bit_order") != "lsb_to_msb":
        raise ValueError("This released checkpoint expects LSB-to-MSB bit order.")
    return config


def resolve_from_root(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def preprocess_image(path: Path, config: dict[str, Any]) -> torch.Tensor:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        image = transform_functional.pil_to_tensor(rgb).to(torch.float32) / 255.0

    if config["center_crop_square"]:
        side = min(image.shape[-2:])
        image = transform_functional.center_crop(image, [side, side])

    image_size = int(config["image_size"])
    image = transform_functional.resize(
        image,
        [image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=bool(config["resize_antialias"]),
    )
    return (image * 255.0).round().clamp(0, 255).to(torch.uint8)


def make_spatial_grid(image_size: int, coordinate_order: str) -> torch.Tensor:
    axis = torch.linspace(-1.0, 1.0, steps=image_size)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if coordinate_order == "yx":
        return torch.stack((yy, xx), dim=-1).reshape(-1, 2)
    if coordinate_order == "xy":
        return torch.stack((xx, yy), dim=-1).reshape(-1, 2)
    raise ValueError("coordinate_order must be either 'xy' or 'yx'.")


def add_bit_coordinates(spatial: torch.Tensor, num_bits: int) -> torch.Tensor:
    bit_axis = torch.linspace(
        -1.0, 1.0, steps=num_bits, device=spatial.device, dtype=spatial.dtype
    )
    spatial = spatial[:, None, :].expand(-1, num_bits, -1)
    bits = bit_axis[None, :, None].expand(spatial.shape[0], -1, 1)
    return torch.cat((spatial, bits), dim=-1)


def decompose_bits(image_chw: torch.Tensor, num_bits: int) -> torch.Tensor:
    values = image_chw.to(torch.int64)
    planes = [(values >> bit) & 1 for bit in range(num_bits)]
    return torch.stack(planes, dim=0).permute(2, 3, 0, 1).contiguous()


def remove_data_parallel_prefix(state_dict: dict[str, torch.Tensor]):
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary containing model_state_dict.")
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_state_dict'.")
    return checkpoint


def build_model(config: dict[str, Any]) -> BitPlaneSiren:
    return BitPlaneSiren(
        hidden_features=int(config["hidden_features"]),
        hidden_layers=int(config["hidden_layers"]),
        out_features=int(config["channels"]),
        num_bits=int(config["num_bits"]),
        first_omega_0=float(config["first_omega_0"]),
        hidden_omega_0=float(config["hidden_omega_0"]),
        bit_index_std=float(config["bit_index_std"]),
        use_db=bool(config["use_db"]),
        use_film_pbd=bool(config["use_film_pbd"]),
    )


@torch.inference_mode()
def predict_bits(
    model: BitPlaneSiren,
    image_size: int,
    num_bits: int,
    coordinate_order: str,
    batch_pixels: int,
    device: torch.device,
    logit_threshold: float,
) -> torch.Tensor:
    spatial = make_spatial_grid(image_size, coordinate_order)
    predictions: list[torch.Tensor] = []
    for start in range(0, spatial.shape[0], batch_pixels):
        pixel_coords = spatial[start : start + batch_pixels].to(device)
        coords = add_bit_coordinates(pixel_coords, num_bits)
        logits = model(coords)
        predictions.append((logits >= logit_threshold).to(torch.uint8).cpu())
    return torch.cat(predictions, dim=0).reshape(
        image_size, image_size, num_bits, 3
    )


def recompose_image(predicted_bits: torch.Tensor) -> np.ndarray:
    num_bits = predicted_bits.shape[2]
    bit_weights = (2 ** torch.arange(num_bits, dtype=torch.int64))[None, None, :, None]
    image = (predicted_bits.to(torch.int64) * bit_weights).sum(dim=2)
    return image.clamp(0, 255).numpy().astype(np.uint8)


def calculate_metrics(
    predicted_bits: torch.Tensor,
    target_bits: torch.Tensor,
    predicted_image: np.ndarray,
    target_image: np.ndarray,
) -> dict[str, Any]:
    bit_errors = int((predicted_bits != target_bits).sum().item())
    total_bits = int(target_bits.numel())
    ber = bit_errors / total_bits

    difference = predicted_image.astype(np.float64) - target_image.astype(np.float64)
    mse = float(np.mean(difference**2))
    rmse = math.sqrt(mse)
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(255.0 / rmse)
    ssim = float(
        structural_similarity(
            target_image,
            predicted_image,
            channel_axis=2,
            data_range=255,
        )
    )
    return {
        "bit_errors": bit_errors,
        "total_bits": total_bits,
        "ber": ber,
        "psnr_db": psnr,
        "ssim": ssim,
        "rmse": rmse,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one SBD-INR checkpoint on the released Kodak image."
    )
    parser.add_argument(
        "--config", default=str(ROOT / "configs" / "kodim23.json")
    )
    parser.add_argument("--image", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-pixels", type=int, default=4096)
    parser.add_argument("--save-reconstruction", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config).resolve())
    image_path = resolve_from_root(args.image or config["image_path"])
    checkpoint_path = resolve_from_root(
        args.checkpoint or config["checkpoint_path"]
    )

    if not image_path.is_file():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Place the released checkpoint at this path or pass --checkpoint."
        )
    if args.batch_pixels < 1:
        raise ValueError("--batch-pixels must be positive.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model = build_model(config).to(device)
    state_dict = remove_data_parallel_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    target_chw = preprocess_image(image_path, config)
    target_bits = decompose_bits(target_chw, int(config["num_bits"]))
    predicted_bits = predict_bits(
        model=model,
        image_size=int(config["image_size"]),
        num_bits=int(config["num_bits"]),
        coordinate_order=str(config["coordinate_order"]),
        batch_pixels=args.batch_pixels,
        device=device,
        logit_threshold=float(config["logit_threshold"]),
    )
    predicted_image = recompose_image(predicted_bits)
    target_image = target_chw.permute(1, 2, 0).numpy()
    metrics = calculate_metrics(
        predicted_bits, target_bits, predicted_image, target_image
    )

    print(f"Image:      {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device:     {device}")
    print(f"Bit errors: {metrics['bit_errors']}/{metrics['total_bits']}")
    print(f"BER:        {metrics['ber']:.12e}")
    print(
        "PSNR:      inf"
        if math.isinf(metrics["psnr_db"])
        else f"PSNR:      {metrics['psnr_db']:.6f} dB"
    )
    print(f"SSIM:       {metrics['ssim']:.6f}")
    print(f"RMSE:       {metrics['rmse']:.6f}")

    if args.save_reconstruction:
        output_path = Path(args.save_reconstruction).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(predicted_image, mode="RGB").save(output_path)
        print(f"Saved:      {output_path}")


if __name__ == "__main__":
    main()
