import glob
import json
import os
import random
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from google.colab import drive
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer, get_cosine_schedule_with_warmup
import numpy as np
from PIL import ImageDraw
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import zipfile
from torch import nn
import open_clip


drive.mount('/content/drive')

import os
import shutil
import zipfile

# === 1. 來源檔案路徑（在雲端硬碟中） ===
drive_zip_path = "/content/drive/MyDrive/public_data/cropped.zip"
drive_json_path = "/content/drive/MyDrive/public_data/train_info.json"
drive_json_path_test = "/content/drive/MyDrive/public_data/test.json"
# === 2. 本機資料夾位置 ===
local_data_dir = "/content/crop"
os.makedirs(local_data_dir, exist_ok=True)

# === 3. 解壓 crop.zip 到本機資料夾 ===
with zipfile.ZipFile(drive_zip_path, 'r') as zip_ref:
    zip_ref.extractall(local_data_dir)
print("已解壓 crop.zip")

# === 4. 複製 JSON 標註檔到本機 ===
shutil.copy(drive_json_path, os.path.join("/content", "train_info.json"))
print("已複製 train_info.json")

drive_json_path_test = "/content/drive/MyDrive/public_data/test.json"
# === 4. 複製 JSON 標註檔到本機 ===
shutil.copy(drive_json_path_test, os.path.join("/content", "test.json"))
print("已複製 test.json")

def seed_everything(seed=2025):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# We provide a sample dataloader for this dataset, you can modify it as needed.
class TextImageDataset(Dataset):
    def __init__(self, data_root, caption_file, tokenizer, size=256):
        self.data_root = data_root
        self.tokenizer = tokenizer
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        self.image_files = glob.glob(os.path.join(data_root, "*.png"))
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]

        ## Load image
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)

        ## Load text prompt
        # Image file name: "mosterID_ACTION_frameID.png"
        # key in "train_ingo.json": "mosterID_ACTION"
        key = img_file.split("/")[-1].split(".")[0]
        key = "_".join(key.split("_")[:-1])

        # Sample caption =  moster description + action description
        given_descriptions = self.captions[key]['given_description']
        given_description = random.choice(given_descriptions)
        caption = f"{given_description} {self.captions[key]['action_description']}"
        caption = "" if random.random() < 0.05 else caption
        inputs = self.tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids

        return {
            "pixel_values": image,
            "input_ids": inputs.squeeze(0),
        }

@torch.no_grad()
def generate_and_save_images(unet, vae, text_encoder, tokenizer, epoch, device, save_folder, guidance_scale=2.0, enable_bar=False):
    unet.eval()
    noise_scheduler = DDPMScheduler()  # 預設值就是你訓練時的
    scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
    scheduler.set_timesteps(10)

    test_prompts = [
        "A red tree monster with a skull face and twisted branches.",
        "Blood-toothed monster with spiked fur, wielding an axe, and wearing armor. The monster is moving.",
        "Gray vulture monster with wings, sharp beak, and trident.",
        "Small, purple fish-like creature with large eye and pink fins. The monster is being hit.",
    ]
    images_list = []
    for i, prompt in enumerate(test_prompts):
        # Conditional and unconditional text embeddings
        cond_ids = tokenizer([prompt], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
        uncond_ids = tokenizer([""], return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
        cond_emb = text_encoder(cond_ids)[0]
        uncond_emb = text_encoder(uncond_ids)[0]

        # Concatenate for classifier-free guidance
        encoder_input = torch.cat([uncond_emb, cond_emb], dim=0)

        latents = torch.randn((1, 4, 32, 32)).to(device)
        scheduler.set_timesteps(10)
        prog_bar = tqdm(scheduler.timesteps) if enable_bar else scheduler.timesteps

        for t in prog_bar:
            # Duplicate latents for both branches
            latent_input = latents.expand(2, -1, -1, -1)  # shape [2, 4, 32, 32]

            noise_pred = unet(latent_input, t.to(device), encoder_hidden_states=encoder_input).sample
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Decode
        latents = latents / 0.18215
        image = vae.decode(latents, return_dict=False)[0]
        image = (image.clamp(-1, 1) + 1) / 2
        image = transforms.ToPILImage()(image[0].cpu())
        images_list.append(image)

    # Display grid
    grid_img = Image.new('RGB', (2 * images_list[0].width, 2 * images_list[0].height))
    for idx, img in enumerate(images_list):
        row, col = divmod(idx, 2)
        grid_img.paste(img, (col * img.width, row * img.height))
    save_path = os.path.join(save_folder, f"epoch_{epoch:03d}.png")
    grid_img.save(save_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img)
    plt.axis('off')
    plt.title(f"Generated Samples Epoch {epoch}")
    plt.show()
    unet.train()

class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 初始化 shadow 權重
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def train():
    seed_everything(2025)
    # ========= Hyperparameters ==========
    train_epochs = 60
    batch_size = 64
    gradient_accumulation_steps = 1
    # You can use gradients accumulation to simulate larger batch size if you have limited GPU memory.
    # Call optimizer.step() every `gradient_accumulation_steps` batches.
    lr = 2e-4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========== Saved folders ==========
    ckpt_folder = "/content/drive/MyDrive/DDIM11/ckpt"
    save_folder = "/content/drive/MyDrive/DDIM11/samples"
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)


    # ========== Load Pretrained Model ==========
    ## You cannot change this part
    pretrain_CLIP_path = "openai/clip-vit-base-patch32"
    pretrain_VAE_path = "CompVis/stable-diffusion-v1-4"

    # Load pre-trained CLIP tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(pretrain_CLIP_path)
    text_encoder = CLIPTextModel.from_pretrained(pretrain_CLIP_path).eval().to(device)
    text_encoder.requires_grad_(False)

    # Load pre-trained VAE
    vae = AutoencoderKL.from_pretrained(pretrain_VAE_path, subfolder="vae").to(device)
    vae.requires_grad_(False)

    # ========== Init ==========
    ## You should modify the model architecture by your self
    unet = UNet2DConditionModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=3,  
    block_out_channels=(192, 384, 768),  
    down_block_types=(
        "CrossAttnDownBlock2D",  
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D"
    ),
    up_block_types=(
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ),
    cross_attention_dim=512,
    attention_head_dim=8  # 強化 attention 深度
    ).to(device)
    unet.train()
    ema = EMA(unet, 0.998)
    optimizer = AdamW(list(unet.parameters()), lr=lr, weight_decay=0.01)

    noise_scheduler = DDPMScheduler()
    noise_scheduler.set_timesteps(1000)

    # ========== Dataset ==========
    dataset = TextImageDataset(
    "/content/crop",
    "/content/train_info.json",
    tokenizer
    )

    print("✅ Dataset initialized")
    print("Dataset size:", len(dataset))

    if len(dataset) == 0:
        print("❌ Dataset is empty. Check your data paths or preprocessing logic.")
    else:
        sample = dataset[0]
        print("First sample keys:", sample.keys())
        print("Text tokens shape:", sample["input_ids"].shape)
        print("Pixel values shape:", sample["pixel_values"].shape)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    print(f"Dataset size: {len(dataset)}")
    print("Number of batches:", len(dataloader))
    # Test the generation pipeline
    num_training_steps = train_epochs * len(dataloader)
    warmup_steps = int(0.05 * num_training_steps)  # 前 5% warmup
    scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
    )

    generate_and_save_images(unet, vae, text_encoder, tokenizer, 0, device, save_folder, guidance_scale=2.0, enable_bar=True)

    # ========== Training ==========
    loss_list = []
    lr_list = []
    loss_accumulated = 0.0
    step = 0
    for epoch in range(train_epochs):
        pbar = tqdm(dataloader)
        for batch in pbar:
            step += 1
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode text and images
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # Scaling

            # Sample noise and timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss.backward()

            clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            ema.update()
            optimizer.zero_grad()
            loss_list.append(loss.item())
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

        current_lr = scheduler.get_last_lr()[0]
        lr_list.append(current_lr)
        print(f"[Epoch {epoch+1}] Learning Rate: {current_lr:.8f}")

        if (10 <= epoch + 1 <= 35) or ((35 < epoch + 1 <= 44) and (epoch + 1) % 2 == 0) or ((44 < epoch + 1 ) and (epoch + 1) % 3 == 0) or (epoch + 1) == train_epochs:
            ema.apply_shadow()
            generate_and_save_images(unet, vae, text_encoder, tokenizer, epoch+1, device, save_folder, guidance_scale=2.0, enable_bar=True)
            torch.save({
            "step": step,
            "epoch": epoch,
            "unet": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "ema": ema.shadow,
            "learning_rate": current_lr  # 額外記錄當下學習率
            }, os.path.join(ckpt_folder, f"checkpoint_epoch{epoch+1}.pt"))
            print(f"[Epoch {epoch+1}] Save Checkpoint")
            ema.restore()

    # ========== Plot Loss Curve After Training ==========
    plt.plot(loss_list)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(1, len(lr_list) + 1), lr_list)
    plt.xlabel("Step")

    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()
