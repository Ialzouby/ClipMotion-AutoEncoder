import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from ClipMotionAE import TextConditionedAutoencoder, pad_or_truncate_motion, MAX_LEN, FEATURE_DIM, BATCH_SIZE, LATENT_DIM, TEXT_DIM

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("="*40)
print(f"✅ Using device: {device}")
print("="*40)

# Load CLIP
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()

def get_clip_text_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        return clip_model(**tokens).last_hidden_state[:, 0, :]

# Dataset wrapper
class HumanML3DTextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        motion = pad_or_truncate_motion(self.ds[idx]["motion"]).view(-1)
        caption = self.ds[idx]["caption"]
        text_emb = get_clip_text_embedding(caption).squeeze(0)
        return motion, text_emb

# Load dataset
print("Loading dataset for validation...")
ds = load_dataset("TeoGchx/HumanML3D", split="train")
dataset = HumanML3DTextDataset(ds)

val_size = int(0.1 * len(dataset))
_, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - val_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load model
model = TextConditionedAutoencoder(input_dim=MAX_LEN * FEATURE_DIM, latent_dim=LATENT_DIM, text_dim=TEXT_DIM).to(device)
model.load_state_dict(torch.load("text_conditioned_autoencoder_humanml3d.pth", map_location=device))
model.eval()

# Metrics
mse_fn = nn.MSELoss(reduction='none')
mae_fn = nn.L1Loss(reduction='none')

total_mse = 0
total_mae = 0
total_l2 = 0
num_samples = 0

with torch.no_grad():
    for motion, text in val_loader:
        motion = motion.to(device)
        text = text.to(device)

        recon, _ = model(motion, text)

        # Element-wise errors
        mse = mse_fn(recon, motion).mean(dim=1)      # (batch_size,)
        mae = mae_fn(recon, motion).mean(dim=1)
        l2 = torch.sqrt(((recon - motion) ** 2).sum(dim=1))  # Euclidean per sample

        total_mse += mse.sum().item()
        total_mae += mae.sum().item()
        total_l2 += l2.sum().item()
        num_samples += motion.size(0)

# Normalize
avg_mse = total_mse / num_samples
avg_mae = total_mae / num_samples
avg_l2 = total_l2 / num_samples

# Report
print(f"\n🎯 Validation Results (CLIP-Conditioned Autoencoder):")
print(f"  ▸ MSE: {avg_mse:.6f}")
print(f"  ▸ MAE: {avg_mae:.6f}")
print(f"  ▸ L2  (Euclidean Distance): {avg_l2:.6f}")
