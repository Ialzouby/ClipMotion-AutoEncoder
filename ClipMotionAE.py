import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
import random

# ---- Config ----
MAX_LEN = 196
FEATURE_DIM = 263
BATCH_SIZE = 32
EPOCHS = 10
LATENT_DIM = 512
TEXT_DIM = 512
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1
LR = 1e-3
SEED = 42

# ---- Setup ----
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Text Encoder ----
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model.eval()

def get_clip_text_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        return text_model(**tokens).last_hidden_state[:, 0, :]  # [CLS] token embedding

# ---- Data Prep ----
def pad_or_truncate_motion(motion, max_len=MAX_LEN, dim=FEATURE_DIM):
    motion = np.array(motion)
    if motion.shape[0] >= max_len:
        return torch.tensor(motion[:max_len], dtype=torch.float32)
    else:
        pad = np.zeros((max_len - motion.shape[0], dim))
        return torch.tensor(np.concatenate([motion, pad]), dtype=torch.float32)

class HumanML3DTextMotionDataset(Dataset):
    def __init__(self, hf_ds):
        self.ds = hf_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        motion = self.ds[idx]["motion"]
        caption = self.ds[idx]["caption"]
        motion_tensor = pad_or_truncate_motion(motion).view(-1)
        text_tensor = get_clip_text_embedding(caption).squeeze(0)
        return motion_tensor, text_tensor

# ---- Dataset + Split ----
print("Loading dataset...")
ds = load_dataset("TeoGchx/HumanML3D", split="train")
dataset = HumanML3DTextMotionDataset(ds)

n = len(dataset)
test_len = int(TEST_SPLIT * n)
val_len = int(VALID_SPLIT * n)
train_len = n - test_len - val_len

train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

# ---- Model ----
class TextConditionedAutoencoder(nn.Module):
    def __init__(self, input_dim=MAX_LEN * FEATURE_DIM, latent_dim=LATENT_DIM, text_dim=TEXT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + text_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_dim)
        )

    def forward(self, motion, text_emb):
        z = self.encoder(motion)
        combined = torch.cat([z, text_emb], dim=1)
        recon = self.decoder(combined)
        return recon, z

model = TextConditionedAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

# ---- Training Loop ----
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    args = parser.parse_args()

    if args.train:
        # ---- Training Loop ----
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0
            for motion, text in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
                motion, text = motion.to(device), text.to(device)
                optimizer.zero_grad()
                recon, _ = model(motion, text)
                loss = loss_fn(recon, motion)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * motion.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for motion, text in val_loader:
                    motion, text = motion.to(device), text.to(device)
                    recon, _ = model(motion, text)
                    loss = loss_fn(recon, motion)
                    val_loss += loss.item() * motion.size(0)
            val_loss /= len(val_loader.dataset)

            print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        torch.save(model.state_dict(), "text_conditioned_autoencoder_humanml3d.pth")
        print("✅ Model saved to text_conditioned_autoencoder_humanml3d.pth")

    else:
        print("⚠️ Skipping training. Use `--train` to enable training mode.")
