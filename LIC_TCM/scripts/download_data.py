#!/usr/bin/env python3
"""
TCM 数据集下载脚本 (Python 版本)
适用于不想用 shell 脚本的情况
"""

import os
import urllib.request
from pathlib import Path


def download_with_progress(url, dest):
    """带进度条的下载函数"""
    import tqdm

    def progress_hook(block_num, block_size, total_size):
        if progress_bar is None:
            return
        progress_bar.update(block_size)

    if os.path.exists(dest):
        print(f"文件已存在，跳过: {dest}")
        return

    print(f"下载: {url} -> {dest}")

    progress_bar = None
    urllib.request.urlretrieve(url, dest, progress_hook)


def download_vimeo90k(data_root):
    """下载 Vimeo-90k Septuplet 数据集"""
    vimeo_dir = Path(data_root) / "vimeo90k"
    vimeo_dir.mkdir(parents=True, exist_ok=True)

    url = "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
    dest = vimeo_dir / "vimeo_septuplet.zip"

    print("\n=== 下载 Vimeo-90k 数据集 ===")
    print("警告: 这是一个约 82GB 的大文件")
    response = input("是否继续? (y/n): ")

    if response.lower() == 'y':
        download_with_progress(url, dest)
        print(f"\n下载完成，请手动解压: unzip {dest}")
    else:
        print("跳过 Vimeo-90k 下载")


def download_kodak(data_root):
    """下载 Kodak 测试集"""
    kodak_dir = Path(data_root) / "kodak"
    kodak_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== 下载 Kodak 测试集 ===")

    for i in range(1, 25):
        url = f"http://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png"
        dest = kodak_dir / f"kodim{i:02d}.png"

        if not dest.exists():
            print(f"下载: kodim{i:02d}.png")
            urllib.request.urlretrieve(url, dest)
        else:
            print(f"已存在: kodim{i:02d}.png")

    print(f"✓ Kodak 测试集下载完成")


def main():
    # 数据根目录
    data_root = os.environ.get("TCM_DATA_ROOT", "/root/autodl-tmp/data")

    print(f"数据将保存到: {data_root}")
    Path(data_root).mkdir(parents=True, exist_ok=True)

    # 下载 Kodak 测试集 (小文件，快速)
    download_kodak(data_root)

    # 下载 Vimeo-90k (大文件，需确认)
    download_vimeo90k(data_root)

    print("\n=== 下载完成 ===")
    print(f"数据目录: {data_root}")
    print("\n下一步:")
    print("1. 解压 Vimeo-90k 数据集")
    print("2. 运行评估脚本验证环境")


if __name__ == "__main__":
    main()
