"""
训练辅助工具函数
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np


class AverageMeter:
    """计算和存储平均值和当前值"""

    def __init__(self, name: str = "", fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """显示训练进度"""

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """
    设置日志

    Args:
        log_dir: 日志目录
        experiment_name: 实验名称

    Returns:
        配置好的 logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: Dict[str, Any],
    filepath: Path,
    is_best: bool = False,
    best_filepath: Path = None,
):
    """
    保存检查点

    Args:
        state: 检查点状态字典
        filepath: 保存路径
        is_best: 是否是最佳模型
        best_filepath: 最佳模型保存路径
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)

    if is_best and best_filepath:
        torch.save(state, best_filepath)


def load_checkpoint(filepath: Path, model, optimizer=None, scheduler=None):
    """
    加载检查点

    Args:
        filepath: 检查点路径
        model: 模型
        optimizer: 优化器 (可选)
        scheduler: 学习率调度器 (可选)

    Returns:
        加载的 epoch 和最佳 loss
    """
    checkpoint = torch.load(filepath, map_location="cpu")

    model.load_state_dict(checkpoint["state_dict"])

    start_epoch = checkpoint.get("epoch", 0)
    best_loss = checkpoint.get("best_loss", float("inf"))

    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return start_epoch, best_loss


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_gpu_memory_usage():
    """获取 GPU 显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        return {
            "allocated": allocated,
            "reserved": reserved,
        }
    return None


class Timer:
    """简单的计时器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.elapsed = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.elapsed = time.time() - self.start_time
        return self.elapsed

    def elapsed_time(self):
        return time.time() - self.start_time
