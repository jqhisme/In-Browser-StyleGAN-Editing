"""Download a reproducible FFHQ subset for encoder INT8 calibration.

Install dependencies: pip install huggingface_hub pillow
Run: python download_ffhq_calibration.py --count 100
"""

import argparse
import random
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image


REPO_ID = "marcosv/ffhq-dataset"
PART1_SIZE = 10_000
OUTPUT_DIR = Path(__file__).resolve().parent / "calibration_faces"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100, help="Number of faces to save (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for repeatable selection")
    args = parser.parse_args()

    if not 1 <= args.count <= PART1_SIZE:
        parser.error(f"--count must be between 1 and {PART1_SIZE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    indices = sorted(random.Random(args.seed).sample(range(PART1_SIZE), args.count))

    for number, index in enumerate(indices, start=1):
        destination = OUTPUT_DIR / f"ffhq_{index:05d}.png"
        if destination.exists():
            print(f"[{number}/{args.count}] Already saved: {destination.name}")
            continue

        source = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f"Part1/{index:05d}.png",
        )
        with Image.open(source) as image:
            image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS).save(destination)
        print(f"[{number}/{args.count}] Saved {destination.name}")

    print(f"Calibration images are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
