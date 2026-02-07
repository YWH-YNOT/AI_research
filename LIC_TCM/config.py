"""
TCM 训练配置文件
集中管理所有训练参数
"""

import argparse
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "tcm_s"  # tcm_s, tcm_m, tcm_l
    N: int = 128  # 潜在通道数
    M: int = 192  # 超先验通道数
    num_heads: int = 8  # 注意力头数
    window_size: int = 8  # 窗口大小


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    seed: int = 42

    # 学习率调度
    scheduler: str = "ReduceLROnPlateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5

    # 梯度裁剪
    clip_grad_norm: float = 1.0

    # 保存频率
    save_freq: int = 10


@dataclass
class DataConfig:
    """数据配置"""
    train_dir: str = "/root/autodl-tmp/data/vimeo90k/vimeo_septuplet"
    test_dir: str = "/root/autodl-tmp/data/kodak"
    patch_size: Tuple[int, int] = (256, 256)
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class CompressionConfig:
    """压缩配置"""
    lambda_mse: float = 0.0130  # MSE 率失真参数
    lambda_ms_ssim: float = 0.0130  # MS-SSIM 率失真参数 (可选)
    metric: str = "mse"  # mse 或 ms-ssim


@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    output_dir: str = "output"
    num_workers: int = 4
    seed: int = 42


# 预设配置
PRESETS = {
    "fast": {  # 快速测试
        "epochs": 10,
        "batch_size": 4,
        "save_freq": 5,
    },
    "baseline": {  # 论文复现
        "epochs": 100,
        "batch_size": 8,
        "save_freq": 10,
    },
    "high_quality": {  # 高质量训练
        "epochs": 200,
        "batch_size": 16,
        "save_freq": 20,
    },
}


def get_config(preset: str = "baseline", **kwargs) -> dict:
    """
    获取配置

    Args:
        preset: 预设名称 (fast, baseline, high_quality)
        **kwargs: 覆盖默认配置的参数

    Returns:
        配置字典
    """
    config = PRESETS.get(preset, PRESETS["baseline"]).copy()
    config.update(kwargs)
    return config


def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description="TCM 训练配置")

    parser.add_argument("--preset", type=str, default="baseline",
                        choices=["fast", "baseline", "high_quality"],
                        help="使用预设配置")

    # 模型
    parser.add_argument("-m", "--model", type=str, default="tcm_s",
                        choices=["tcm_s", "tcm_m", "tcm_l"])
    parser.add_argument("--lambda", type=float, dest="lambda_val",
                        default=0.0130)

    # 训练
    parser.add_argument("-e", "--epochs", type=int)
    parser.add_argument("-b", "--batch-size", type=int)
    parser.add_argument("--lr", type=float)

    # 数据
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--test-dataset", type=str)

    # 系统
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--log-dir", type=str)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true", dest="use_cpu")

    return parser.parse_args()


if __name__ == "__main__":
    # 打印默认配置
    args = parse_args()
    print("默认配置:")
    print(f"  模型: {args.model}")
    print(f"  Lambda: {args.lambda_val}")
    print(f"  预设: {args.preset}")
