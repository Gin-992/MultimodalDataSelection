import os
import json
import argparse
from tqdm import tqdm
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

def compute_similarity(model, vis_processors, txt_processors, image, caption, device):
    # Preprocess inputs
    img_tensor = vis_processors["eval"](image).unsqueeze(0).to(device)
    text_input = txt_processors["eval"](caption)
    # Extract projected features
    feats_img = model.extract_features({"image": img_tensor, "text_input": [text_input]}, mode="image").image_embeds_proj
    feats_txt = model.extract_features({"image": img_tensor, "text_input": [text_input]}, mode="text").text_embeds_proj
    # Use first token (CLS) embedding for each
    img_feat = feats_img[:, 0, :]
    txt_feat = feats_txt[:, 0, :]
    # Compute cosine similarity
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
    sim = (img_feat @ txt_feat.t()).item()
    return sim

def main(args):
    # Prepare device
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    # Load model and preprocessors
    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip_feature_extractor", model_type="base", is_eval=True, device=device
    )

    # Read input JSON
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Process each entry
    results = []
    for entry in tqdm(data, desc="Computing similarities"):
        img_rel = entry.get("image", "")
        img_path = os.path.join(args.image_dir, img_rel)
        if not os.path.isfile(img_path):
            print(f"Warning: image file not found: {img_path}")
            continue
        conv = entry.get("conversations", [])
        text = " \n".join([c.get("value", "") for c in conv]).strip()
        try:
            image = Image.open(img_path).convert("RGB")
            sim_score = compute_similarity(model, vis_processors, txt_processors, image, text, device)
            entry["similarity_score"] = sim_score
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Output results
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)
        print(f"Results written to {args.output_json}")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))

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
    parser = argparse.ArgumentParser(
        description="Compute image-text similarity using BLIP feature extractor"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Specify the model for the image text similarity task"
    )
    parser.add_argument(
        "--json_path", type=str, required=True,
        help="Path to input JSON file containing image paths and captions"
    )
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Directory where image files are stored"
    )
    parser.add_argument(
        "--output_json", type=str, default=None,
        help="Optional path to write output JSON with similarity scores"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run inference on (e.g., cuda or cpu)"
    )
    parser.add_argument("--top_output_json", help="Optional: write top-percent JSON")
    parser.add_argument("--top_percent", type=float, default=10.0,
                        help="Percentage of top entries to select (default: 10)")
    args = parser.parse_args()
    main(args)
