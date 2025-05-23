import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import json
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from baselines.models.RDE import build_model

# Simulated args
args = types.SimpleNamespace()
args.pretrain_choice = 'ViT-B/32'
args.img_size = 224
args.stride_size = 16
args.temperature = 0.07
args.select_ratio = 0.5
args.loss_names = 'TAL'
args.tau = 0.05
args.margin = 0.3
args.json_path = "your_input.json"
args.image_dir = "your_image_dir"

# Simulated base model builder and embedding layers
def build_CLIP_from_openai_pretrained(pretrain_choice, img_size, stride_size):
    class DummyBase(nn.Module):
        def encode_image(self, x):
            return torch.randn(x.size(0), 1, 512), None
        def encode_text(self, x):
            return torch.randn(x.size(0), 77, 512), None
    return DummyBase(), {'embed_dim': 512}

class VisualEmbeddingLayer(nn.Module):
    def __init__(self, ratio=0.5): super().__init__()
    def forward(self, x, attn): return torch.mean(x, dim=1)

class TexualEmbeddingLayer(nn.Module):
    def __init__(self, ratio=0.5): super().__init__()
    def forward(self, x, ids, attn): return torch.mean(x, dim=1)

# Model definition
class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text)
        return x[:, 0, :].float()

# Build model
model = build_model(args, num_classes=11003)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

# Load text tokenizer (simulated tokenization)
def tokenize_text(text):
    tokens = torch.randint(0, 10000, (1, 77))  # Replace with real tokenizer
    return tokens

with open(args.json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for entry in tqdm(data):
    img_rel = entry.get("image", "")
    img_path = os.path.join(args.image_dir, img_rel)
    try:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Warning: cannot open {img_path}: {e}")
        entry["similarity_score"] = None
        continue

    conv = entry.get("conversations", [])
    text = " \n".join([c.get("value", "") for c in conv]).strip()
    text_tensor = tokenize_text(text)

    with torch.no_grad():
        img_feat = model.encode_image(image_tensor)
        txt_feat = model.encode_text(text_tensor)

        img_feat = F.normalize(img_feat, dim=1)
        txt_feat = F.normalize(txt_feat, dim=1)

        similarity = (txt_feat @ img_feat.T).item()
        entry["similarity_score"] = round(similarity, 4)
        print(f"{img_rel} similarity: {similarity:.4f}")
