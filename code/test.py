from types import SimpleNamespace
import os
import open_clip
import glob
import json
import torch
import re
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from typing import List, Tuple
from scipy import linalg
from PIL import Image
from torchvision.models import inception_v3, Inception_V3_Weights
from torch import nn
"""## DDPM"""

@torch.no_grad()
def generate_batch(prompts, unet, vae, tokenizer, text_encoder, device, steps=50, guidance_scale=7.5):
    unet.eval()
    noise_scheduler = DDPMScheduler()
    scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
    scheduler.set_timesteps(steps)

    batch_size = len(prompts)
    cond_ids = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
    uncond_ids = tokenizer([""] * batch_size, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)

    cond_emb = text_encoder(cond_ids)[0]
    uncond_emb = text_encoder(uncond_ids)[0]
    encoder_input = torch.cat([uncond_emb, cond_emb], dim=0)  # (2B, 77, D)

    latents = torch.randn((batch_size, 4, 32, 32)).to(device)

    for t in scheduler.timesteps:
        latent_input = latents.repeat(2, 1, 1, 1)
        noise_pred = unet(latent_input, t.to(device), encoder_hidden_states=encoder_input).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / 0.18215
    images = vae.decode(latents, return_dict=False)[0]
    images = (images.clamp(-1, 1) + 1) / 2

    pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
    return pil_images

def test_inference(ckpt_path, test_json_path, save_folder, device="cuda", steps=50, guidance_scale=7.5, batch_size=4):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    vae.requires_grad_(False)

    print(f"🔍 Loading checkpoint: {ckpt_path}")
    # 1. 初始化 unet（使用原始 config）
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
    attention_head_dim=8
    ).to(device)

    # 2. 載入訓練好的權重（推薦用 EMA）
    ckpt = torch.load(ckpt_path, map_location=device)
    if "ema" in ckpt:
        print("✅ Using EMA weights")
        unet.load_state_dict(ckpt["ema"])
    else:
        print("⚠️ Using raw UNet weights")
        unet.load_state_dict(ckpt["unet"])

    os.makedirs(save_folder, exist_ok=True)
    with open(test_json_path, "r") as f:
        test_data = list(json.load(f).items())

    print(f"🚀 Start batch inference: {len(test_data)} samples | Batch size={batch_size}")
    for i in tqdm(range(0, len(test_data), batch_size), desc="Batch generating"):
        batch_items = test_data[i:i+batch_size]
        prompts = [item[1]["text_prompt"] for item in batch_items]
        filenames = [item[1]["image_name"] for item in batch_items]

        images = generate_batch(prompts, unet, vae, tokenizer, text_encoder, device,
                                steps=steps, guidance_scale=guidance_scale)
        for img, fname in zip(images, filenames):
            img.save(os.path.join(save_folder, fname))

class ImageSizeError(ValueError):
    pass

class FIDDataset(Dataset):
    def __init__(self, image_paths: str, image_size: int = None):
        self.image_paths = image_paths
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        if self.image_size is not None and image.size != (self.image_size, self.image_size):
            raise ImageSizeError(
                f"Image {image_path} has size {image.size}, expected "
                f"{self.image_size}x{self.image_size}"
            )

        if self.transform:
            image = self.transform(image)
        return image


class CLIPDataset(Dataset):
    def __init__(self, base2path: str, items: List[Tuple[str, str]], preprocess, tokenizer, image_size: int = None):
        self.base2path = base2path
        self.items = items
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.image_size = image_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        base_name = self.items[idx]["image_name"]
        image_path = self.base2path[base_name]
        image = Image.open(image_path).convert('RGB')
        if self.image_size is not None and image.size != (self.image_size, self.image_size):
            raise ImageSizeError(
                f"Image {image_path} has size {image.size}, expected "
                f"{self.image_size}x{self.image_size}."
            )
        image = self.preprocess(image)

        text_prompt = self.items[idx]["text_prompt"]
        tokens = self.tokenizer(text_prompt)[0]

        return image, tokens


def get_inception_model() -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.eval()
    return model

def get_inception_feature(dataloader: DataLoader, model: nn.Module, device: torch.device, verbose: bool = False) -> torch.Tensor:
    features_list = []
    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            ncols=0,
            desc="Extracting features",
            leave=False,
            disable=not verbose
        ):
            batch = batch.to(device)
            features = model(batch)
            features_list.append(features.cpu())
    return torch.cat(features_list, axis=0)


def calculate_fid(mu1, sigma1, mu2, sigma2):
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


class CLIPScore:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def get_image_embedding(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_embedding(self, tokens):
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def clip_score_image_image(self, images1, images2):
        embs1 = self.get_image_embedding(images1)
        embs2 = self.get_image_embedding(images2)
        scores = (embs1 * embs2).sum(dim=-1)
        return scores

    def clip_score_image_text(self, images, tokens):
        embsi = self.get_image_embedding(images)
        embst = self.get_text_embedding(tokens)
        scores = (embsi * embst).sum(dim=-1)
        return scores


def run_fid(args, device):
    # ================ Load reference statistics ================
    # 改成直接用參數指定的絕對路徑
    real_mu = np.load(args.ref_mu_path)
    real_sigma = np.load(args.ref_sigma_path)

    # ================ List and check images ================
    image_paths = glob.glob(os.path.join(args.fake_img_root, '**', '*.png'), recursive=True)
    if args.num_images is not None and len(image_paths) != args.num_images:
        raise ValueError(
            f"Expected {args.num_images} PNG files in zip file, "
            f"found {len(image_paths)}.")
    if args.verbose:
        print(f"[INFO] Found {len(image_paths)} images in the zip file.")

    # ================ Load and process images =================
    dataset = FIDDataset(image_paths, args.image_size)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    # Initialize the Inception model and extract features
    model = get_inception_model().to(device)
    if args.verbose:
        print("[INFO] Generating Inception model features...")
    features = get_inception_feature(dataloader, model, device, verbose=args.verbose)
    del model
    features = features.numpy()
    fake_mu = np.mean(features, axis=0)
    fake_sigma = np.cov(features, rowvar=False)

    # ================ Calculate FID =================
    if args.verbose:
        print("[INFO] Calculating FID score...")
    fid = calculate_fid(real_mu, real_sigma, fake_mu, fake_sigma)
    if args.verbose:
        print(f"[INFO] FID score: {fid:.4f}.")
    return {"FID": fid}


def get_image_base2path(image_root, items, num_images=None, verbose=False):
    image_paths = glob.glob(os.path.join(image_root, '**', '*.png'), recursive=True)
    image_base2path = {os.path.basename(path): path for path in image_paths}
    if len(image_paths) != len(items):
        raise ValueError(
            f"Expected {len(items)} PNG files in zip file, found "
            f"{len(image_paths)}.")
    if num_images is None:
        num_images = len(items)
    else:
        num_images = num_images
    if len(image_base2path) != num_images:
        raise ValueError(
            f"The number of unique image filenames ({len(image_base2path)}) "
            f"does not match the number of items in the test JSON file ({num_images})."
        )
    if verbose:
        print(f"[INFO] Found {len(image_paths)} images in the zip file.")

    if verbose:
        print(
            "[INFO] Checking if all image paths from the test JSON file are "
            "present in the zip file.")
    for item in items:
        # item["image_name"] is the base filename of the image
        if item["image_name"] not in image_base2path:
            raise ValueError(
                f"Image path {item['image_name']} from the test JSON file "
                f"not found in the zip file."
            )
    return image_base2path


def run_clip(args, device):
    if not ('clip_i' in args.scores or 'clip_t' in args.scores):
        raise ValueError(
            "CLIP scores can only be calculated if 'clip_i' or 'clip_t' is in "
            "the --scores argument."
        )

    # ================ Load reference statistics ================
    if not os.path.exists(args.test_json_path):
        raise FileNotFoundError(f"Test JSON file {args.test_json_path} does not exist.")
    if args.verbose:
        print(f"[INFO] Loading test JSON file: {args.test_json_path}.")
    with open(args.test_json_path, 'r') as f:
        items = json.load(f)

    items = [
        {
            "image_name": item["image_name"],
            "text_prompt": item["text_prompt"]
        } for item in items.values()
    ]

    # ================ List and check images ================

    fake_image_base2path = get_image_base2path(
        args.fake_img_root,
        items,
        num_images=args.num_images,
        verbose=args.verbose
    )

    # ================ Load CLIP model ================
    if args.verbose:
        print(f"[INFO] Loading CLIP model: {args.model_name} with pretrained weights: {args.pretrained}.")
    model, _, preprocess = open_clip.create_model_and_transforms(args.model_name, pretrained=args.pretrained)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model_name)
    clip_score = CLIPScore(model, tokenizer)

    # ================ Load and process images =================

    fake_dataset = CLIPDataset(
        fake_image_base2path, items, preprocess, tokenizer, image_size=args.image_size
    )
    fake_dataloader = DataLoader(
        fake_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    if args.verbose:
        print(f"[INFO] Calculating CLIP scores for {len(fake_dataset)} images...")
    scores_list = defaultdict(list)
    for fake_image, tokens in tqdm(
        fake_dataloader):
        fake_image = fake_image.to(device)
        tokens = tokens.to(device)
        for score in args.scores:
             if score == 'clip_t':
                 scores = clip_score.clip_score_image_text(fake_image, tokens)
             else:
                 scores = None
             if scores is not None:
                 scores_list[score].append(scores.cpu())


    # ================ Save the score =================
    output_json = {}
    for score in args.scores:
        if score == 'clip_i':
            score_name = "CLIP Image-Image Score"
        elif score == 'clip_t':
            score_name = "CLIP Image-Text Score"
        else:
            score_name = None
        if score_name is not None:
            if args.verbose:
                mean_score = torch.cat(scores_list[score]).mean().item()
                print(f"{score_name}: {mean_score:.6f}")
            output_json[score_name] = mean_score

    return output_json

"""### step=15, guidance=3"""

# 要測試的 epoch 清單
target_epochs = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,40,42]

# 固定參數設定
test_json_path = "/content/test.json"
ref_mu_path = "/content/drive/MyDrive/public_data/score/test_mu.npy"
ref_sigma_path = "/content/drive/MyDrive/public_data/score/test_sigma.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_list = []
os.makedirs("/content/generated_eval4", exist_ok=True)
for epoch in target_epochs:
    ckpt_path = f"/content/drive/MyDrive/DDIM11/ckpt/checkpoint_epoch{epoch}.pt"
    save_folder = f"/content/generated_test11_{epoch}_15_3_2"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🎯 Epoch {epoch}: Generating images...")

    # 推論產圖
    test_inference(
        ckpt_path=ckpt_path,
        test_json_path=test_json_path,
        save_folder=save_folder,
        steps=15,
        guidance_scale=3,
        batch_size=32
    )

    # 設定評分參數
    args = SimpleNamespace(
        ref_mu_path=ref_mu_path,
        ref_sigma_path=ref_sigma_path,
        test_json_path=test_json_path,
        fake_img_root=save_folder,
        image_size=256,
        num_images=None,
        scores=["fid", "clip_t"],
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        batch_size=32,
        num_workers=2,
        verbose=True,
        output_path=f"/content/generated_eval4/scores_ep{epoch}.json"
    )

    # 執行評分
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        output_json.update(run_clip(args, device))
    if 'fid' in args.scores:
        output_json.update(run_fid(args, device))

    # 印出分數
    print(f"✅ Epoch {epoch} scores:")
    for key in ["FID", "CLIP Image-Text Score"]:
        if key in output_json:
            print(f"  {key}: {output_json[key]:.4f}")

    # 儲存分數
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # 收集結果
    result_list.append({
        "epoch": epoch,
        "FID": output_json.get("FID", None),
        "CLIP": output_json.get("CLIP Image-Text Score", None)
    })

# 總結
print("\n📊 Summary:")
for r in result_list:
    print(f"Epoch {r['epoch']:>2} | FID: {r['FID']:.4f} | CLIP: {r['CLIP']:.4f}")

# 評分資料夾路徑
eval_dir = "/content/generated_eval4"

# 讀取所有 scores_ep*.json
result_list = []
for fname in sorted(os.listdir(eval_dir)):
    match = re.match(r"scores_ep(\d+)\.json", fname)
    if match:
        epoch = int(match.group(1))
        with open(os.path.join(eval_dir, fname), "r") as f:
            data = json.load(f)
            result_list.append({
                "epoch": epoch,
                "FID": data.get("FID", None),
                "CLIP": data.get("CLIP Image-Text Score", None)
            })

# 照 epoch 排序
result_list.sort(key=lambda x: x["epoch"])

# 提取資料
epochs = [r["epoch"] for r in result_list]
fid_scores = [r["FID"] for r in result_list]
clip_scores = [r["CLIP"] for r in result_list]

# 畫 FID vs. Epoch
plt.figure()
plt.plot(epochs, fid_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.gca().invert_yaxis()  # FID 越低越好
plt.show()

# 畫 CLIP Score vs. Epoch
plt.figure()
plt.plot(epochs, clip_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("CLIP Image-Text Score")
plt.title("CLIP Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.show()

#=================================== Another Try ==========================================
"""### step=10, guidance=2"""

# 要測試的 epoch 清單
target_epochs = [10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,40,42,44,45,48,51,54,57,60]

# 固定參數設定
test_json_path = "/content/test.json"
ref_mu_path = "/content/drive/MyDrive/public_data/score/test_mu.npy"
ref_sigma_path = "/content/drive/MyDrive/public_data/score/test_sigma.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_list = []
os.makedirs("/content/generated_eval1", exist_ok=True)
for epoch in target_epochs:
    ckpt_path = f"/content/drive/MyDrive/DDIM11/ckpt/checkpoint_epoch{epoch}.pt"
    save_folder = f"/content/drive/MyDrive/generated_test11_{epoch}_10_2_2"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🎯 Epoch {epoch}: Generating images...")

    # 推論產圖
    test_inference(
        ckpt_path=ckpt_path,
        test_json_path=test_json_path,
        save_folder=save_folder,
        steps=10,
        guidance_scale=2,
        batch_size=32
    )

    # 設定評分參數
    args = SimpleNamespace(
        ref_mu_path=ref_mu_path,
        ref_sigma_path=ref_sigma_path,
        test_json_path=test_json_path,
        fake_img_root=save_folder,
        image_size=256,
        num_images=None,
        scores=["fid", "clip_t"],
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        batch_size=32,
        num_workers=2,
        verbose=True,
        output_path=f"/content/generated_eval1/scores_ep{epoch}.json"
    )

    # 執行評分
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        output_json.update(run_clip(args, device))
    if 'fid' in args.scores:
        output_json.update(run_fid(args, device))

    # 印出分數
    print(f"✅ Epoch {epoch} scores:")
    for key in ["FID", "CLIP Image-Text Score"]:
        if key in output_json:
            print(f"  {key}: {output_json[key]:.4f}")

    # 儲存分數
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # 收集結果
    result_list.append({
        "epoch": epoch,
        "FID": output_json.get("FID", None),
        "CLIP": output_json.get("CLIP Image-Text Score", None)
    })

# 總結
print("\n📊 Summary:")
for r in result_list:
    print(f"Epoch {r['epoch']:>2} | FID: {r['FID']:.4f} | CLIP: {r['CLIP']:.4f}")

# 要測試的 epoch 清單
target_epochs = [18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,40,42,44,45,48,51,54,57,60]

# 固定參數設定
test_json_path = "/content/test.json"
ref_mu_path = "/content/drive/MyDrive/public_data/score/test_mu.npy"
ref_sigma_path = "/content/drive/MyDrive/public_data/score/test_sigma.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_list = []
for epoch in target_epochs:
    ckpt_path = f"/content/drive/MyDrive/DDIM11/ckpt/checkpoint_epoch{epoch}.pt"
    save_folder = f"/content/generated_test11_{epoch}_10_2_2"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🎯 Epoch {epoch}: Generating images...")

    # 推論產圖
    test_inference(
        ckpt_path=ckpt_path,
        test_json_path=test_json_path,
        save_folder=save_folder,
        steps=10,
        guidance_scale=2,
        batch_size=32
    )

    # 設定評分參數
    args = SimpleNamespace(
        ref_mu_path=ref_mu_path,
        ref_sigma_path=ref_sigma_path,
        test_json_path=test_json_path,
        fake_img_root=save_folder,
        image_size=256,
        num_images=None,
        scores=["fid", "clip_t"],
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        batch_size=32,
        num_workers=2,
        verbose=True,
        output_path=f"/content/generated_eval1/scores_ep{epoch}.json"
    )

    # 執行評分
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        output_json.update(run_clip(args, device))
    if 'fid' in args.scores:
        output_json.update(run_fid(args, device))

    # 印出分數
    print(f"✅ Epoch {epoch} scores:")
    for key in ["FID", "CLIP Image-Text Score"]:
        if key in output_json:
            print(f"  {key}: {output_json[key]:.4f}")

    # 儲存分數
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # 收集結果
    result_list.append({
        "epoch": epoch,
        "FID": output_json.get("FID", None),
        "CLIP": output_json.get("CLIP Image-Text Score", None)
    })

# 總結
print("\n📊 Summary:")
for r in result_list:
    print(f"Epoch {r['epoch']:>2} | FID: {r['FID']:.4f} | CLIP: {r['CLIP']:.4f}")

# 評分資料夾路徑
eval_dir = "/content/generated_eval1"

# 讀取所有 scores_ep*.json
result_list = []
for fname in sorted(os.listdir(eval_dir)):
    match = re.match(r"scores_ep(\d+)\.json", fname)
    if match:
        epoch = int(match.group(1))
        with open(os.path.join(eval_dir, fname), "r") as f:
            data = json.load(f)
            result_list.append({
                "epoch": epoch,
                "FID": data.get("FID", None),
                "CLIP": data.get("CLIP Image-Text Score", None)
            })

# 照 epoch 排序
result_list.sort(key=lambda x: x["epoch"])

# 提取資料
epochs = [r["epoch"] for r in result_list]
fid_scores = [r["FID"] for r in result_list]
clip_scores = [r["CLIP"] for r in result_list]

# 畫 FID vs. Epoch
plt.figure()
plt.plot(epochs, fid_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.gca().invert_yaxis()  # FID 越低越好
plt.show()

# 畫 CLIP Score vs. Epoch
plt.figure()
plt.plot(epochs, clip_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("CLIP Image-Text Score")
plt.title("CLIP Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.show()

"""### step=10, guidance=3"""

# 要測試的 epoch 清單
target_epochs = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,40,42]

# 固定參數設定
test_json_path = "/content/test.json"
ref_mu_path = "/content/drive/MyDrive/public_data/score/test_mu.npy"
ref_sigma_path = "/content/drive/MyDrive/public_data/score/test_sigma.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_list = []
os.makedirs("/content/generated_eval2", exist_ok=True)
for epoch in target_epochs:
    ckpt_path = f"/content/drive/MyDrive/DDIM11/ckpt/checkpoint_epoch{epoch}.pt"
    save_folder = f"/content/generated_test11_{epoch}_10_3_2"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🎯 Epoch {epoch}: Generating images...")

    # 推論產圖
    test_inference(
        ckpt_path=ckpt_path,
        test_json_path=test_json_path,
        save_folder=save_folder,
        steps=10,
        guidance_scale=3,
        batch_size=32
    )

    # 設定評分參數
    args = SimpleNamespace(
        ref_mu_path=ref_mu_path,
        ref_sigma_path=ref_sigma_path,
        test_json_path=test_json_path,
        fake_img_root=save_folder,
        image_size=256,
        num_images=None,
        scores=["fid", "clip_t"],
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        batch_size=32,
        num_workers=2,
        verbose=True,
        output_path=f"/content/generated_eval2/scores_ep{epoch}.json"
    )

    # 執行評分
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        output_json.update(run_clip(args, device))
    if 'fid' in args.scores:
        output_json.update(run_fid(args, device))

    # 印出分數
    print(f"✅ Epoch {epoch} scores:")
    for key in ["FID", "CLIP Image-Text Score"]:
        if key in output_json:
            print(f"  {key}: {output_json[key]:.4f}")

    # 儲存分數
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # 收集結果
    result_list.append({
        "epoch": epoch,
        "FID": output_json.get("FID", None),
        "CLIP": output_json.get("CLIP Image-Text Score", None)
    })

# 總結
print("\n📊 Summary:")
for r in result_list:
    print(f"Epoch {r['epoch']:>2} | FID: {r['FID']:.4f} | CLIP: {r['CLIP']:.4f}")

# 評分資料夾路徑
eval_dir = "/content/generated_eval2"

# 讀取所有 scores_ep*.json
result_list = []
for fname in sorted(os.listdir(eval_dir)):
    match = re.match(r"scores_ep(\d+)\.json", fname)
    if match:
        epoch = int(match.group(1))
        with open(os.path.join(eval_dir, fname), "r") as f:
            data = json.load(f)
            result_list.append({
                "epoch": epoch,
                "FID": data.get("FID", None),
                "CLIP": data.get("CLIP Image-Text Score", None)
            })

# 照 epoch 排序
result_list.sort(key=lambda x: x["epoch"])

# 提取資料
epochs = [r["epoch"] for r in result_list]
fid_scores = [r["FID"] for r in result_list]
clip_scores = [r["CLIP"] for r in result_list]

# 畫 FID vs. Epoch
plt.figure()
plt.plot(epochs, fid_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.gca().invert_yaxis()  # FID 越低越好
plt.show()

# 畫 CLIP Score vs. Epoch
plt.figure()
plt.plot(epochs, clip_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("CLIP Image-Text Score")
plt.title("CLIP Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.show()

"""### step=10, guidance=4"""

# 要測試的 epoch 清單
target_epochs = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,40,42]

# 固定參數設定
test_json_path = "/content/test.json"
ref_mu_path = "/content/drive/MyDrive/public_data/score/test_mu.npy"
ref_sigma_path = "/content/drive/MyDrive/public_data/score/test_sigma.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_list = []
os.makedirs("/content/generated_eval3", exist_ok=True)
for epoch in target_epochs:
    ckpt_path = f"/content/drive/MyDrive/DDIM11/ckpt/checkpoint_epoch{epoch}.pt"
    save_folder = f"/content/generated_test11_{epoch}_10_4_2"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🎯 Epoch {epoch}: Generating images...")

    # 推論產圖
    test_inference(
        ckpt_path=ckpt_path,
        test_json_path=test_json_path,
        save_folder=save_folder,
        steps=10,
        guidance_scale=4,
        batch_size=32
    )

    # 設定評分參數
    args = SimpleNamespace(
        ref_mu_path=ref_mu_path,
        ref_sigma_path=ref_sigma_path,
        test_json_path=test_json_path,
        fake_img_root=save_folder,
        image_size=256,
        num_images=None,
        scores=["fid", "clip_t"],
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        batch_size=32,
        num_workers=2,
        verbose=True,
        output_path=f"/content/generated_eval2/scores_ep{epoch}.json"
    )

    # 執行評分
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        output_json.update(run_clip(args, device))
    if 'fid' in args.scores:
        output_json.update(run_fid(args, device))

    # 印出分數
    print(f"✅ Epoch {epoch} scores:")
    for key in ["FID", "CLIP Image-Text Score"]:
        if key in output_json:
            print(f"  {key}: {output_json[key]:.4f}")

    # 儲存分數
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # 收集結果
    result_list.append({
        "epoch": epoch,
        "FID": output_json.get("FID", None),
        "CLIP": output_json.get("CLIP Image-Text Score", None)
    })

# 總結
print("\n📊 Summary:")
for r in result_list:
    print(f"Epoch {r['epoch']:>2} | FID: {r['FID']:.4f} | CLIP: {r['CLIP']:.4f}")

# 評分資料夾路徑
eval_dir = "/content/generated_eval2"

# 讀取所有 scores_ep*.json
result_list = []
for fname in sorted(os.listdir(eval_dir)):
    match = re.match(r"scores_ep(\d+)\.json", fname)
    if match:
        epoch = int(match.group(1))
        with open(os.path.join(eval_dir, fname), "r") as f:
            data = json.load(f)
            result_list.append({
                "epoch": epoch,
                "FID": data.get("FID", None),
                "CLIP": data.get("CLIP Image-Text Score", None)
            })

# 照 epoch 排序
result_list.sort(key=lambda x: x["epoch"])

# 提取資料
epochs = [r["epoch"] for r in result_list]
fid_scores = [r["FID"] for r in result_list]
clip_scores = [r["CLIP"] for r in result_list]

# 畫 FID vs. Epoch
plt.figure()
plt.plot(epochs, fid_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.gca().invert_yaxis()  # FID 越低越好
plt.show()

# 畫 CLIP Score vs. Epoch
plt.figure()
plt.plot(epochs, clip_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("CLIP Image-Text Score")
plt.title("CLIP Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.show()

"""### step=20, guidance=3"""

# 要測試的 epoch 清單
target_epochs = [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,38,40,42]
# 固定參數設定
test_json_path = "/content/test.json"
ref_mu_path = "/content/drive/MyDrive/public_data/score/test_mu.npy"
ref_sigma_path = "/content/drive/MyDrive/public_data/score/test_sigma.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_list = []
os.makedirs("/content/generated_eval5", exist_ok=True)
for epoch in target_epochs:
    ckpt_path = f"/content/drive/MyDrive/DDIM11/ckpt/checkpoint_epoch{epoch}.pt"
    save_folder = f"/content/generated_test11_{epoch}_20_3_2"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🎯 Epoch {epoch}: Generating images...")

    # 推論產圖
    test_inference(
        ckpt_path=ckpt_path,
        test_json_path=test_json_path,
        save_folder=save_folder,
        steps=20,
        guidance_scale=3,
        batch_size=32
    )

    # 設定評分參數
    args = SimpleNamespace(
        ref_mu_path=ref_mu_path,
        ref_sigma_path=ref_sigma_path,
        test_json_path=test_json_path,
        fake_img_root=save_folder,
        image_size=256,
        num_images=None,
        scores=["fid", "clip_t"],
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        batch_size=32,
        num_workers=2,
        verbose=True,
        output_path=f"/content/generated_eval5/scores_ep{epoch}.json"
    )

    # 執行評分
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        output_json.update(run_clip(args, device))
    if 'fid' in args.scores:
        output_json.update(run_fid(args, device))

    # 印出分數
    print(f"✅ Epoch {epoch} scores:")
    for key in ["FID", "CLIP Image-Text Score"]:
        if key in output_json:
            print(f"  {key}: {output_json[key]:.4f}")

    # 儲存分數
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # 收集結果
    result_list.append({
        "epoch": epoch,
        "FID": output_json.get("FID", None),
        "CLIP": output_json.get("CLIP Image-Text Score", None)
    })

# 總結
print("\n📊 Summary:")
for r in result_list:
    print(f"Epoch {r['epoch']:>2} | FID: {r['FID']:.4f} | CLIP: {r['CLIP']:.4f}")

# 評分資料夾路徑
eval_dir = "/content/generated_eval5"

# 讀取所有 scores_ep*.json
result_list = []
for fname in sorted(os.listdir(eval_dir)):
    match = re.match(r"scores_ep(\d+)\.json", fname)
    if match:
        epoch = int(match.group(1))
        with open(os.path.join(eval_dir, fname), "r") as f:
            data = json.load(f)
            result_list.append({
                "epoch": epoch,
                "FID": data.get("FID", None),
                "CLIP": data.get("CLIP Image-Text Score", None)
            })

# 照 epoch 排序
result_list.sort(key=lambda x: x["epoch"])

# 提取資料
epochs = [r["epoch"] for r in result_list]
fid_scores = [r["FID"] for r in result_list]
clip_scores = [r["CLIP"] for r in result_list]

# 畫 FID vs. Epoch
plt.figure()
plt.plot(epochs, fid_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.gca().invert_yaxis()  # FID 越低越好
plt.show()

# 畫 CLIP Score vs. Epoch
plt.figure()
plt.plot(epochs, clip_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("CLIP Image-Text Score")
plt.title("CLIP Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.show()

"""## DDIM"""

@torch.no_grad()
def generate_batch(prompts, unet, vae, tokenizer, text_encoder, device, steps=50, guidance_scale=7.5):
    unet.eval()
    scheduler = DDIMScheduler()
    scheduler.set_timesteps(steps)
    scheduler.eta = 0.0

    batch_size = len(prompts)
    cond_ids = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)
    uncond_ids = tokenizer([""] * batch_size, return_tensors="pt", padding="max_length", truncation=True, max_length=77).input_ids.to(device)

    cond_emb = text_encoder(cond_ids)[0]
    uncond_emb = text_encoder(uncond_ids)[0]
    encoder_input = torch.cat([uncond_emb, cond_emb], dim=0)  # (2B, 77, D)

    latents = torch.randn((batch_size, 4, 32, 32)).to(device)

    for t in scheduler.timesteps:
        latent_input = latents.repeat(2, 1, 1, 1)
        noise_pred = unet(latent_input, t.to(device), encoder_hidden_states=encoder_input).sample
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / 0.18215
    images = vae.decode(latents, return_dict=False)[0]
    images = (images.clamp(-1, 1) + 1) / 2

    pil_images = [transforms.ToPILImage()(img.cpu()) for img in images]
    return pil_images

def test_inference(ckpt_path, test_json_path, save_folder, device="cuda", steps=50, guidance_scale=7.5, batch_size=4):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").eval().to(device)
    text_encoder.requires_grad_(False)
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(device)
    vae.requires_grad_(False)

    print(f"🔍 Loading checkpoint: {ckpt_path}")
    # 1. 初始化 unet（使用原始 config）
    # 本地路徑，不需要再用 from_config()，而是直接用 from_pretrained()
    # 1. 建立一樣的 UNet 架構
    unet = UNet2DConditionModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    block_out_channels=(192, 384, 768, 768),
    layers_per_block=2,
    down_block_types=(
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"
    ),
    up_block_types=(
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"
    ),
    cross_attention_dim=512,
    attention_head_dim=8
    ).to(device)

    # 2. 載入訓練好的權重（推薦用 EMA）
    ckpt = torch.load(ckpt_path, map_location=device)
    if "ema" in ckpt:
        print("✅ Using EMA weights")
        unet.load_state_dict(ckpt["ema"])
    else:
        print("⚠️ Using raw UNet weights")
        unet.load_state_dict(ckpt["unet"])

    os.makedirs(save_folder, exist_ok=True)
    with open(test_json_path, "r") as f:
        test_data = list(json.load(f).items())  # 轉成 list 才能 batch

    print(f"🚀 Start batch inference: {len(test_data)} samples | Batch size={batch_size}")
    for i in tqdm(range(0, len(test_data), batch_size), desc="Batch generating"):
        batch_items = test_data[i:i+batch_size]
        prompts = [item[1]["text_prompt"] for item in batch_items]
        filenames = [item[1]["image_name"] for item in batch_items]

        images = generate_batch(prompts, unet, vae, tokenizer, text_encoder, device,
                                steps=steps, guidance_scale=guidance_scale)
        for img, fname in zip(images, filenames):
            img.save(os.path.join(save_folder, fname))

"""### Step=10, guidance=2"""

# 要測試的 epoch 清單（每兩個存一個）
target_epochs = list(range(6, 61, 4))

# 固定參數設定
test_json_path = "/content/test.json"
ref_mu_path = "/content/drive/MyDrive/public_data/score/test_mu.npy"
ref_sigma_path = "/content/drive/MyDrive/public_data/score/test_sigma.npy"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

result_list = []
os.makedirs("/content/generated_eval1", exist_ok=True)
for epoch in target_epochs:
    ckpt_path = f"/content/drive/MyDrive/DDIM11/ckpt/checkpoint_epoch{epoch}.pt"
    save_folder = f"/content/generated_test6_{epoch}_10_2_2"
    os.makedirs(save_folder, exist_ok=True)

    print(f"\n🎯 Epoch {epoch}: Generating images...")

    # 推論產圖
    test_inference(
        ckpt_path=ckpt_path,
        test_json_path=test_json_path,
        save_folder=save_folder,
        steps=10,
        guidance_scale=2,
        batch_size=16
    )

    # 設定評分參數
    args = SimpleNamespace(
        ref_mu_path=ref_mu_path,
        ref_sigma_path=ref_sigma_path,
        test_json_path=test_json_path,
        fake_img_root=save_folder,
        image_size=256,
        num_images=None,
        scores=["fid", "clip_t"],
        model_name="ViT-B-32-quickgelu",
        pretrained="openai",
        batch_size=32,
        num_workers=2,
        verbose=True,
        output_path=f"/content/generated_eval1/scores_ep{epoch}.json"
    )

    # 執行評分
    output_json = {}
    if 'clip_i' in args.scores or 'clip_t' in args.scores:
        output_json.update(run_clip(args, device))
    if 'fid' in args.scores:
        output_json.update(run_fid(args, device))

    # 印出分數
    print(f"✅ Epoch {epoch} scores:")
    for key in ["FID", "CLIP Image-Text Score"]:
        if key in output_json:
            print(f"  {key}: {output_json[key]:.4f}")

    # 儲存分數
    with open(args.output_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # 收集結果
    result_list.append({
        "epoch": epoch,
        "FID": output_json.get("FID", None),
        "CLIP": output_json.get("CLIP Image-Text Score", None)
    })

# 總結
print("\n📊 Summary:")
for r in result_list:
    print(f"Epoch {r['epoch']:>2} | FID: {r['FID']:.4f} | CLIP: {r['CLIP']:.4f}")

import os
import re
import json
import matplotlib.pyplot as plt

# 評分資料夾路徑
eval_dir = "/content/generated_eval1"

# 讀取所有 scores_ep*.json
result_list = []
for fname in sorted(os.listdir(eval_dir)):
    match = re.match(r"scores_ep(\d+)\.json", fname)
    if match:
        epoch = int(match.group(1))
        with open(os.path.join(eval_dir, fname), "r") as f:
            data = json.load(f)
            result_list.append({
                "epoch": epoch,
                "FID": data.get("FID", None),
                "CLIP": data.get("CLIP Image-Text Score", None)
            })

# 照 epoch 排序
result_list.sort(key=lambda x: x["epoch"])

# 提取資料
epochs = [r["epoch"] for r in result_list]
fid_scores = [r["FID"] for r in result_list]
clip_scores = [r["CLIP"] for r in result_list]

# 畫 FID vs. Epoch
plt.figure()
plt.plot(epochs, fid_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("FID")
plt.title("FID Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.gca().invert_yaxis()  # FID 越低越好
plt.show()

# 畫 CLIP Score vs. Epoch
plt.figure()
plt.plot(epochs, clip_scores, marker='o')
plt.xlabel("Epoch")
plt.ylabel("CLIP Image-Text Score")
plt.title("CLIP Score vs. Epoch")
plt.grid(True)
plt.xticks(epochs)
plt.show()

import shutil

# 將 generated_images 資料夾壓縮成 zip
shutil.make_archive("generated_test6_20_10_2_2", "zip", "generated_test6_20_10_2_2")

print("✅ 壓縮完成，可下載 generated_images.zip 檔案")
from google.colab import files

files.download("/content/generated_test6_20_10_2_2.zip")
print("✅ 預測完成，generated_images.zip 已儲存並下載。")