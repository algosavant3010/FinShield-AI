"""
scripts/setup_data.py
Run this ONCE after downloading your datasets.
It scans your video folders and writes data/manifest.json

Usage:
    python scripts/setup_data.py \
        --ff_real   /data/ff++/real \
        --ff_fake   /data/ff++/fake \
        --dfdc_real /data/dfdc/real \
        --dfdc_fake /data/dfdc/fake \
        --output    data/manifest.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_loader import generate_manifest


def main():
    parser = argparse.ArgumentParser(description="Generate dataset manifest for FinShield AI")
    parser.add_argument("--ff_real",    type=str, default=None, help="Path to FF++ real videos")
    parser.add_argument("--ff_fake",    type=str, default=None, help="Path to FF++ fake videos")
    parser.add_argument("--dfdc_real",  type=str, default=None, help="Path to DFDC real videos")
    parser.add_argument("--dfdc_fake",  type=str, default=None, help="Path to DFDC fake videos")
    parser.add_argument("--celeb_real", type=str, default=None, help="Path to Celeb-DF real videos")
    parser.add_argument("--celeb_fake", type=str, default=None, help="Path to Celeb-DF fake videos")
    parser.add_argument("--output",     type=str, default="data/manifest.json")
    parser.add_argument("--val_ratio",  type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()

    # Build dataset_roots from provided paths
    dataset_roots = {}
    if args.ff_real and args.ff_fake:
        dataset_roots["FF++"] = {"real": args.ff_real, "fake": args.ff_fake}
    if args.dfdc_real and args.dfdc_fake:
        dataset_roots["DFDC"] = {"real": args.dfdc_real, "fake": args.dfdc_fake}
    if args.celeb_real and args.celeb_fake:
        dataset_roots["CelebDF"] = {"real": args.celeb_real, "fake": args.celeb_fake}

    if not dataset_roots:
        print("ERROR: No dataset paths provided. Use --ff_real, --ff_fake etc.")
        print("Example:")
        print("  python scripts/setup_data.py --ff_real /data/ff++/real --ff_fake /data/ff++/fake")
        sys.exit(1)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate_manifest(dataset_roots, args.output, args.val_ratio, args.test_ratio)
    print(f"\nDone! Now run: python train.py --manifest {args.output}")


if __name__ == "__main__":
    main()
