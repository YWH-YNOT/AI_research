import argparse
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# === 关键导入 ===
from custom_data import Vimeo90kSeptuplet          # 加载自定义数据
from models.tcm import TCM_S, TCM_M, TCM_L         # 加载自定义模型

class AverageMeter:
    """计算和存储平均值和当前值"""
    def __init__(self):
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

class RateDistortionLoss(nn.Module):
    """自定义率失真损失函数"""
    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["mse_loss"] = self.mse(output["x_hat"], target)
        
        # 计算 Bpp (Bits per pixel)
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        # 最终 Loss = Lambda * D + R
        out["loss"] = self.lmbda * 255**2 * out["mse_loss"] + out["bpp_loss"]

        return out

def configure_optimizers(net, args):
    """
    配置两个优化器：
    1. model_optimizer: 优化网络权重 (卷积层等)
    2. aux_optimizer: 优化熵模型的 CDF 参数 (Quantization)
    这是 CompressAI 训练的标准流程，必不可少。
    """
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
    (params_dict[n] for n in sorted(parameters)),
    lr=args.learning_rate,  # <--- 改成这样
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=1e-3,
    )
    return optimizer, aux_optimizer

def train_one_epoch(model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer):
    model.train()
    device = next(model.parameters()).device

    loss_meter = AverageMeter()
    bpp_meter = AverageMeter()
    mse_meter = AverageMeter()

    start_time = time.time()

    # 这里的 batch 是一个列表，包含 7 个 tensor (im1 到 im7)
    # 形状为: [Batch, 3, H, W]
    for i, images_list in enumerate(train_dataloader):
        # 策略：Vimeo90k 是 7 帧视频，我们这里训练的是图像压缩模型
        # 所以我们随机选其中一帧来训练，增加数据的多样性
        idx = random.randint(0, 6) 
        d = images_list[idx].to(device)

        # 1. 优化模型主参数
        optimizer.zero_grad()
        out_net = model(d)
        
        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        # 2. 优化辅助参数 (Auxiliary Loss) - 必须有这一步！
        aux_loss = model.aux_loss()
        aux_optimizer.zero_grad()
        aux_loss.backward()
        aux_optimizer.step()

        # 记录数据
        loss_meter.update(out_criterion["loss"].item())
        bpp_meter.update(out_criterion["bpp_loss"].item())
        mse_meter.update(out_criterion["mse_loss"].item())

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: [{i}/{len(train_dataloader)}]\t"
                f"Loss: {loss_meter.avg:.3f} | "
                f"MSE: {mse_meter.avg:.5f} | "
                f"Bpp: {bpp_meter.avg:.4f}"
            )
            
            # 记录到 Tensorboard
            current_step = epoch * len(train_dataloader) + i
            writer.add_scalar('Train/Loss', loss_meter.avg, current_step)
            writer.add_scalar('Train/Bpp', bpp_meter.avg, current_step)
            writer.add_scalar('Train/MSE', mse_meter.avg, current_step)

    print(f"Epoch {epoch} done. Time: {time.time() - start_time:.2f}s")

def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("checkpoint", "checkpoint_best"))

def parse_args(argv):
    parser = argparse.ArgumentParser(description="TCM Training Script")
    # 模型参数
    parser.add_argument("-m", "--model", type=str, default="tcm_s", choices=["tcm_s", "tcm_m", "tcm_l"], help="Model architecture")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset path")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of epochs")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=0.0130, help="Bit-rate distortion parameter")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--patch-size", type=int, nargs=2, default=(256, 256), help="Image patch size")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--seed", type=float, default=100, help="Random seed")
    parser.add_argument("--clip_max_norm", default=1.0, type=float, help="Gradient clipping max norm")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-freq", type=int, default=10, help="Save frequency")
    
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    
    # 随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # 1. 设置 Transforms (关键修正：必须做 Crop 和 ToTensor)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(args.patch_size),
        transforms.ToTensor()
    ])

    # 2. 加载数据集
    train_dataset = Vimeo90kSeptuplet(
        args.dataset, 
        split='train', 
        transform=train_transforms
    )
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Dataset length: {len(train_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    # 3. 创建模型 (关键修正：使用我们的 custom models)
    model_cls = {
        "tcm_s": TCM_S,
        "tcm_m": TCM_M,
        "tcm_l": TCM_L,
    }
    net = model_cls[args.model]()
    net = net.to(device)

    # 4. 设置优化器 (主参数 + 辅助参数)
    optimizer, aux_optimizer = configure_optimizers(net, args)
    
    # 5. 定义 Loss
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    # 6. 加载断点 (如果有)
    last_epoch = 0
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    # 7. 创建日志目录
    log_dir = Path("logs") / f"{args.model}_L{args.lmbda}"
    writer = SummaryWriter(log_dir=log_dir)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # === 开始训练循环 ===
    print(f"Start training: Model={args.model}, Lambda={args.lmbda}, Batch={args.batch_size}")
    
    for epoch in range(last_epoch, args.epochs):
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer
        )

        # 保存模型
        if args.save:
            is_best = False # 这里简单处理，你可以加逻辑判断 best loss
            if epoch % args.save_freq == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": 0, # 这里可以填 loss_meter.avg
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": None,
                    },
                    is_best,
                    filename=str(checkpoint_dir / f"checkpoint_{args.model}_{epoch}.pth.tar")
                )

if __name__ == "__main__":
    main(sys.argv[1:])