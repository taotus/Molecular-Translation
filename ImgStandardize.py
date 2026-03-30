import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
import os
import concurrent.futures
from tqdm import tqdm


def get_image_path(img_id: str, train: bool = True) -> str:
    if train:
        path = f"mol_img/train/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.png"
    else:
        path = f"mol_img/test/{img_id[0]}/{img_id[1]}/{img_id[2]}/{img_id}.png"
    return path


def compute_folder_stats(folder_path):
    """
    计算单个文件夹内所有图片的像素和、平方和、像素总数（递归处理子文件夹）
    返回 (pixel_sum, pixel_sq_sum, total_pixels, processed_count)
    """
    pixel_sum = 0.
    pixel_sq_sum = 0.
    total_pixels = 0
    processed = 0
    error_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if not file.lower().endswith('.png'):
                continue
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    gray = img.convert('L')
                    arr = np.array(gray, dtype=np.float32)
                    pixel_sum += arr.sum()
                    pixel_sq_sum += (arr ** 2).sum()
                    total_pixels += arr.size
                    processed += 1
                    if processed % 100 == 0:
                        print(f"处理完成第 {processed} 张图片")
            except Exception as e:
                error_files.append((file_path, str(e)))
                print(f"读取失败: {file_path} - {e}")

    if error_files:
        print(f"文件夹{folder_path} 共有 {len(error_files)} 个文件读取失败")

    return pixel_sum, pixel_sq_sum, total_pixels, processed

def compute_global_stats(root_dir, num_workers=8):
    """
    并行计算整个训练集或测试集的统计量
    root_dir: 顶层目录，如 "mol_img/train"
    num_workers: 并行进程数
    """
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"目录不存在: {root_dir}")

    # 获取所有第一层子目录（如 0,1,2,...,f）
    try:
        first_level_dirs = [d for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d))]
        print(f"子目录: {first_level_dirs}")
    except PermissionError as e:
        print(f"无法访问目录: {root_dir} - {e}")
        raise

    if not first_level_dirs:
        print(f"{root_dir} 下没有子文件夹")
        return None, None, 0

    # 并行处理每个第一层子文件夹
    total_sum = 0.0
    total_sq_sum = 0.0
    total_pixels = 0
    total_images = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(compute_folder_stats, os.path.join(root_dir, d)): d for d in first_level_dirs}
        with tqdm(total=len(first_level_dirs), desc="处理子文件夹") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    s, sq, px, cnt = future.result()
                    total_sum += s
                    total_sq_sum += sq
                    total_pixels += px
                    total_images += cnt
                except Exception as e:
                    print(f"处理子文件夹时出错: {futures[future]} - {e}")
                pbar.update(1)

    if total_pixels == 0:
        print("未找到任何有效图片，无法计算统计量")
        return None, None, 0

    mean = total_sum / total_pixels
    # 计算标准差：sqrt(E(x^2) - (E(x))^2)
    std = np.sqrt(total_sq_sum / total_pixels - mean ** 2)

    print(f"处理完成: 共处理 {total_images} 张图片，总像素数 {total_pixels}")
    print(f"全局均值: {mean:.4f}, 全局标准差: {std:.4f}")

    return mean, std, total_images

def standardize_images(img_tensor, mean, std) -> torch.Tensor:
    img_tensor -= mean
    img_tensor /= std
    return - img_tensor

def pad_images(imgs, pad_value=0):
    """
    将一批图像填充到相同尺寸（取 batch 中最大高、宽），返回堆叠的张量。

    Args:
        images: list of torch.Tensor，每个形状为  (H, W)（灰度）
        pad_value: 填充值，标量或元组（每个通道一个值）

    Returns:
        torch.Tensor: 形状为 (B, C, max_H, max_W) 的填充后 batch
    """

    # 获取最大高宽
    max_h = max(img.size(0) for img in imgs)
    max_w = max(img.size(1) for img in imgs)

    padded = []
    for img in imgs:
        h, w = img.size(0), img.size(1)
        # 需要填充的尺寸 (top, bottom, left, right)
        pad_top = 0
        pad_bottom = max_h - h
        pad_left = 0
        pad_right = max_w - w
        # 注意 F.pad 的 padding 顺序是 (left, right, top, bottom)
        padded_img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=pad_value)
        padded.append(padded_img)

    return torch.stack(padded, dim=0).unsqueeze(1)

def image2tensor(img_id, train: bool=True):
    img = torch.tensor(
        np.array(Image.open(get_image_path(img_id, train)).convert('L')),
        dtype=torch.float16
    )
    return img

def process_images(img_ids, mean, std, pad: int=0, train: bool=True) -> torch.Tensor:
    imgs = []
    for idx in img_ids:
        imgs.append(image2tensor(idx, train))
    padded_imgs = pad_images(imgs, pad)
    img_tensor = standardize_images(padded_imgs, mean, std)

    return img_tensor

if __name__ == "__main__":
    train_root = f"mol_img/train"

    mean, std, num_imgs = compute_global_stats(train_root)

    if mean is not None:
        with open("std_record.txt", mode="w") as file:
            file.write(f"mean: {mean}\n"
                       f"std: {std}\n"
                       f"num_imgs: {num_imgs}")




