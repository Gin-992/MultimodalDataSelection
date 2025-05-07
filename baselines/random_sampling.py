import os
import json
import argparse
import random
import shutil
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Randomly sample entries from a JSON dataset (and optionally copy images)"
    )
    parser.add_argument(
        "--json_path", "-j",
        required=True,
        help="Path to the input JSON file (list of objects)"
    )
    parser.add_argument(
        "--output_json", "-o",
        required=True,
        help="Path to write the sampled JSON"
    )
    parser.add_argument(
        "--sample_size", "-n",
        type=int,
        default=50000,
        help="Number of entries to sample (default: 50000)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--image_dir",
        help="Base directory where images live (required if copying images)"
    )
    parser.add_argument(
        "--output_image_dir",
        help="Directory to copy sampled images into (mirrors JSON paths)"
    )
    return parser.parse_args()

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def copy_images(sampled, image_dir, out_dir):
    for entry in tqdm(sampled, desc="Copying images"):
        rel = entry.get("image", "").strip()
        if not rel:
            continue
        src = os.path.join(image_dir, rel)
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"Warning: failed to copy {src} → {dst}: {e}")

def main():
    args = parse_args()
    if args.output_image_dir and not args.image_dir:
        raise ValueError("`--image_dir` is required if `--output_image_dir` is set")

    # set seed
    if args.seed is not None:
        random.seed(args.seed)

    # load full dataset
    data = load_json(args.json_path)
    total = len(data)
    if args.sample_size > total:
        raise ValueError(f"Sample size {args.sample_size} exceeds dataset size {total}")

    # sample without replacement
    sampled = random.sample(data, args.sample_size)

    # save sampled JSON
    save_json(sampled, args.output_json)
    print(f"Saved {len(sampled)} entries to {args.output_json}")

    # optional: copy images
    if args.output_image_dir:
        os.makedirs(args.output_image_dir, exist_ok=True)
        copy_images(sampled, args.image_dir, args.output_image_dir)
        print(f"Copied images to {args.output_image_dir}")

if __name__ == "__main__":
    main()
