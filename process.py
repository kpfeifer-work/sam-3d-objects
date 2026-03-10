# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys
import argparse
import os

# import inference code
sys.path.append("notebook")
from inference import Inference, load_image, load_mask


def default_mask_path(image_path: str) -> str:
    base, _ext = os.path.splitext(image_path)
    return f"{base}_mask.png"


def default_output_path(image_path: str) -> str:
    base, _ext = os.path.splitext(image_path)
    return f"{base}.glb"


def main():
    parser = argparse.ArgumentParser(description="Process image with SAM 3D Objects")

    # positional image path
    parser.add_argument("image", help="Path to the input image file (RGB/RGBA)")

    # optional mask + output + seed
    parser.add_argument("--mask", default=None, help="Path to mask image. Default: <image>_mask.png")
    parser.add_argument("--output", default=None, help="Output .glb path. Default: <image>.glb")
    parser.add_argument("--seed", type=int, default=42, help="Random seed controlling variation")

    # config overrides
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="OmegaConf override(s), e.g. --set slat_cfg_strength=2 "
             "--set ss_preprocessor.img_transform.transforms.1.size=640",
    )

    # optionally enable compile (note: can increase VRAM / startup time)
    parser.add_argument("--compile", action="store_true", help="Enable torch compile (may increase VRAM)")

    args = parser.parse_args()

    # resolve paths
    mask_path = args.mask or default_mask_path(args.image)
    out_path = args.output or default_output_path(args.image)

    print(f"Loading image: {args.image}")
    image = load_image(args.image)

    print(f"Loading mask:  {mask_path}")
    mask = load_mask(mask_path)

    # load model
    tag = "hf"
    config_path = f"checkpoints/{tag}/pipeline.yaml"

    # IMPORTANT: pass overrides list (args.set is always a list now)
    inference = Inference(config_path, compile=args.compile, overrides=args.set)

    # run model
    output = inference(image, mask, seed=args.seed)

    # export mesh
    output["glb"].export(out_path)
    print(f"Saved mesh as {out_path}")


if __name__ == "__main__":
    main()