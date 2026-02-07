#!/usr/bin/env python3
"""
TCM 模型评估脚本
用于在 Kodak/Tecnick 测试集上评估模型性能
"""

import argparse
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="TCM 模型评估")

    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="预训练模型路径")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="测试集目录路径")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="重建图像保存目录")
    parser.add_argument("--cuda", action="store_true",
                        help="使用 CUDA")

    return parser.parse_args()


def compute_psnr(img1, img2):
    """计算 PSNR"""
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ms_ssim(img1, img2):
    """计算 MS-SSIM (简化版，完整版需要 pytorch_msssim 库)"""
    # 这里仅作为占位符
    from compressai.metrics import ms_ssim
    return ms_ssim(img1, img2)


def load_image(image_path):
    """加载图像"""
    img = Image.open(image_path).convert("RGB")
    img = np.array(img).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1)  # C x H x W
    return img.unsqueeze(0)  # 添加 batch 维度


def save_image(tensor, save_path):
    """保存图像"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def evaluate(model, dataset_dir, device, output_dir=None):
    """评估模型"""
    dataset_dir = Path(dataset_dir)
    image_files = sorted(dataset_dir.glob("*.png")) + sorted(dataset_dir.glob("*.jpg"))

    if not image_files:
        print(f"错误: 在 {dataset_dir} 中没有找到图像文件")
        return

    print(f"找到 {len(image_files)} 张测试图像")

    model.eval()

    metrics = defaultdict(list)

    with torch.no_grad():
        for img_path in tqdm(image_files, desc="评估中"):
            # 加载图像
            img = load_image(img_path).to(device)

            # 前向传播
            start_time = time.time()
            output = model(img)
            encode_time = time.time() - start_time

            reconstructed = output["x_hat"]

            # 计算 BPP
            num_pixels = img.size(0) * img.size(2) * img.size(3)
            bpp = sum(
                (torch.log(likelihoods).sum() / (-np.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            ).item()

            # 计算 PSNR
            psnr = compute_psnr(reconstructed, img)

            # 计算 MS-SSIM
            try:
                ms_ssim_val = compute_ms_ssim(reconstructed, img)
            except:
                ms_ssim_val = 0.0

            # 记录指标
            metrics["psnr"].append(psnr)
            metrics["bpp"].append(bpp)
            metrics["ms_ssim"].append(ms_ssim_val.item() if torch.is_tensor(ms_ssim_val) else ms_ssim_val)
            metrics["encode_time"].append(encode_time)

            # 保存重建图像
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                save_path = output_dir / img_path.name
                save_image(reconstructed, save_path)

    # 打印结果
    print("\n" + "=" * 60)
    print("评估结果 (平均):")
    print("=" * 60)
    print(f"PSNR:    {np.mean(metrics['psnr']):.4f} dB")
    print(f"BPP:     {np.mean(metrics['bpp']):.4f}")
    print(f"MS-SSIM: {np.mean(metrics['ms_ssim']):.4f}")
    print(f"编码时间: {np.mean(metrics['encode_time'])*1000:.2f} ms")
    print("=" * 60)

    return metrics


def main():
    args = parse_args()

    # 设备
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # 尝试获取模型架构
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 根据模型类型创建模型
    # 这里需要根据实际保存的模型类型来创建
    from compressai.models import BMSHJ2018Factorized

    # 简化处理：使用默认模型
    # 实际应根据模型参数选择正确的模型
    model = BMSHJ2018Factorized(quality=3, pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 评估
    metrics = evaluate(model, args.dataset, device, args.output)

    # 保存结果
    if args.output:
        import json
        output_dir = Path(args.output)
        results_file = output_dir / "metrics.json"

        with open(results_file, "w") as f:
            json.dump({
                "psnr": float(np.mean(metrics["psnr"])),
                "bpp": float(np.mean(metrics["bpp"])),
                "ms_ssim": float(np.mean(metrics["ms_ssim"])),
                "encode_time": float(np.mean(metrics["encode_time"])),
            }, f, indent=2)

        print(f"结果已保存到: {results_file}")


if __name__ == "__main__":
    main()
