import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from datasets import load_dataset
from transformers import CLIPTokenizer, CLIPTextModel
from ClipMotionAE import pad_or_truncate_motion, TextConditionedAutoencoder

# ====== Constants ======
MAX_LEN = 196
FEATURE_DIM = 263
NUM_JOINTS = 22
JOINT_DIM = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== SMPL-H kinematic chain ======
SMPLH_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15, 0, 17, 18, 19, 20]
BONES = [(i, p) for i, p in enumerate(SMPLH_PARENTS) if p != -1]

# ====== Load model ======
model = TextConditionedAutoencoder().to(device)
model.load_state_dict(torch.load("text_conditioned_autoencoder_humanml3d.pth", map_location=device))
model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
text_model.eval()

def get_clip_text_embedding(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        return text_model(**tokens).last_hidden_state[:, 0, :]

def motion_to_xyz(motion_tensor):
    motion = motion_tensor.reshape(MAX_LEN, FEATURE_DIM)
    xyz = motion[:, :NUM_JOINTS * JOINT_DIM].reshape(MAX_LEN, NUM_JOINTS, JOINT_DIM)
    return xyz

# ====== Plot Helper ======
def plot_frame(ax, joints, bones, color, title):
    ax.clear()
    for i, j in bones:
        ax.plot(*zip(joints[i], joints[j]), c=color)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c=color, s=10)
    ax.set_title(title)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(0, 2.5)
    ax.view_init(elev=110, azim=-90)
    ax.axis('off')

# ====== Load Data Sample ======
ds = load_dataset("TeoGchx/HumanML3D", split="train")
sample = ds[0]
caption = sample["caption"]

motion_tensor = pad_or_truncate_motion(sample["motion"]).view(-1).unsqueeze(0).to(device)
text_emb = get_clip_text_embedding(caption).to(device)

# ====== Reconstruct ======
with torch.no_grad():
    recon_tensor, _ = model(motion_tensor, text_emb)

# ====== Convert to XYZ ======
original_xyz = motion_to_xyz(motion_tensor.squeeze(0).cpu())
recon_xyz = motion_to_xyz(recon_tensor.squeeze(0).cpu())

# ====== Generate Frames ======
os.makedirs("frames", exist_ok=True)
frames = []

for t in range(MAX_LEN):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    plot_frame(ax1, original_xyz[t], BONES, 'blue', 'Original')
    plot_frame(ax2, recon_xyz[t], BONES, 'red', 'Reconstruction')

    fname = f"frames/frame_{t:03d}.png"
    plt.savefig(fname)
    frames.append(imageio.imread(fname))
    plt.close()

# ====== Save Video ======
imageio.mimsave("reconstruction_comparison.mp4", frames, fps=15)
print("✅ Saved to reconstruction_comparison.mp4")

# Optional cleanup
for f in os.listdir("frames"):
    os.remove(os.path.join("frames", f))
os.rmdir("frames")
