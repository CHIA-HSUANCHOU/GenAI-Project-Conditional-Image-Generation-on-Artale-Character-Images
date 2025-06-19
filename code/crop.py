import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import zipfile

input_dir = "train"      # 改為你本機的資料夾路徑
output_dir = "crop"     # 本機儲存裁切後圖片

def get_crop_center_and_size(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        h, w = image_np.shape[:2]
        return w // 2, h // 2, min(w, h)
    x, y, w, h = cv2.boundingRect(coords)
    cx, cy = x + w // 2, y + h // 2
    size = max(w, h)
    return cx, cy, size

def crop_square_around_center(image_np, cx, cy, size):
    h, w = image_np.shape[:2]
    half = size // 2
    x_start = max(0, cx - half)
    y_start = max(0, cy - half)
    x_end = min(w, cx + half)
    y_end = min(h, cy + half)
    cropped = image_np[y_start:y_end, x_start:x_end]
    result = np.ones((size, size, 3), dtype=np.uint8) * 255
    ph, pw = cropped.shape[:2]
    result[:ph, :pw] = cropped
    return result

def process_one_image(fname):
    img_path = os.path.join(input_dir, fname)
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        cx, cy, crop_size = get_crop_center_and_size(img_np)
        cropped = crop_square_around_center(img_np, cx, cy, crop_size)
        resized = cv2.resize(cropped, (256, 256), interpolation=cv2.INTER_AREA)

        save_path = os.path.join(output_dir, fname)
        Image.fromarray(resized).save(save_path)
        return True
    except Exception as e:
        return f"{fname} failed: {e}"

if __name__ == "__main__":
    all_png_files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(".png")]

    print(f"🔧 開始處理 {len(all_png_files)} 張圖片...")
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap_unordered(process_one_image, all_png_files), total=len(all_png_files)))

    failures = [r for r in results if r is not True]
    if failures:
        print(f"\n⚠️ 共 {len(failures)} 張圖片處理失敗：")
        for f in failures:
            print(f)
    else:
        print("✅ 所有圖片處理成功")

    zip_filename = "cropped.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_dir)
                    zipf.write(file_path, arcname)

    print(f"\n📦 壓縮完成：{zip_filename}")