import os
import json
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

def compute_similarity(image: Image.Image, text: str, processor, model, device) -> float:
    """
    Compute cosine‐based similarity between an image and a text string using CLIP.
    """
    inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image
        probs = F.softmax(logits, dim=1)
    return probs[0, 0].item()

def main():
    # ─── Argument Parsing ─────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Compute CLIP similarity and optionally extract top-percent entries"
    )
    parser.add_argument("--json_path",      required=True, help="Input JSON file path")
    parser.add_argument("--image_dir",      required=True, help="Directory of images")
    parser.add_argument("--output_json",    required=True, help="Output JSON with scores")
    parser.add_argument("--top_output_json",         help="Optional: write top-percent JSON")
    parser.add_argument("--top_percent",   type=float, default=10.0,
                        help="Percentage of top entries to select (default: 10)")  # :contentReference[oaicite:5]{index=5}
    args = parser.parse_args()

    # ─── Load CLIP Model ───────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)
    model     = CLIPModel.from_pretrained(model_name).to(device)

    # ─── Read and Enrich JSON ─────────────────────────────────────────
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in tqdm(data):
        img_rel = entry.get("image", "")
        img_path = os.path.join(args.image_dir, img_rel)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: cannot open {img_path}: {e}")
            entry["similarity_score"] = None
            continue

        conv = entry.get("conversations", [])
        text = " \n".join([c.get("value", "") for c in conv]).strip()
        score = compute_similarity(image, text, processor, model, device)
        entry["similarity_score"] = score

    # ─── Save Full Results ─────────────────────────────────────────────
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # ─── Optional: Extract Top-Percent Entries ────────────────────────
    if args.top_output_json:
        # Filter out invalid scores
        valid = [e for e in data if isinstance(e.get("similarity_score"), (int, float))]
        # Sort descending by score :contentReference[oaicite:6]{index=6}
        sorted_list = sorted(valid, key=lambda e: e["similarity_score"], reverse=True)
        # Determine number of top entries (at least 1) :contentReference[oaicite:7]{index=7}
        n_top = max(int(len(sorted_list) * args.top_percent / 100.0), 1)
        top_entries = sorted_list[:n_top]
        # Write top-percent JSON
        with open(args.top_output_json, "w", encoding="utf-8") as f:
            json.dump(top_entries, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
