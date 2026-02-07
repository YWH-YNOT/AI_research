import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import math
import glob
import time

# =========================================================
# 关键修改：告诉 Python 去 'models' 文件夹里找 tcm
# =========================================================
try:
    from models.tcm import TCM
except ImportError:
    # 备用方案：万一你在 models 目录下运行
    try:
        from tcm import TCM
    except ImportError:
        print("❌ 严重错误：找不到 tcm.py！")
        print("请确认你的目录结构是：")
        print("  - LIC_TCM/")
        print("    - eval.py")
        print("    - models/")
        print("      - tcm.py")
        exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="TCM 模型评估脚本")
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="模型权重路径 (.pth.tar)")
    parser.add_argument("-d", "--data", type=str, required=True, help="测试图片文件夹路径")
    parser.add_argument("--cuda", action="store_true", default=True, help="使用 GPU")
    return parser.parse_args()

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    if mse == 0:
        return 100
    return -10 * math.log10(mse)

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out_net['likelihoods'].values()).item()

def main():
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    print(f"🔄 正在初始化模型 (TCM-Small)...")
    # 初始化 TCM 模型 (Small 版本配置)
    model = TCM(config=[2, 2, 2, 2, 2, 2], head=[8, 16, 32, 32, 16, 8])
    model = model.to(device)
    model.eval()

    # 加载权重
    print(f"📂 正在加载权重: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 兼容处理：检查是否有 'state_dict' 键
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        
    # 加载参数
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # 尝试去除 module. 前缀 (多卡训练常见问题)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    print("✅ 模型加载成功！开始评估...")

    # 准备图片变换
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 获取所有图片 (递归查找)
    # 支持 png 和 jpg
    img_paths = glob.glob(os.path.join(args.data, "*.png")) + \
                glob.glob(os.path.join(args.data, "*.jpg"))
    
    if not img_paths:
        print(f"⚠️  警告：在路径 {args.data} 下没有找到 .png 或 .jpg 图片！")
        return

    # 统计指标
    total_psnr = 0
    total_bpp = 0
    count = 0
    
    with torch.no_grad():
        for img_path in img_paths:
            # 读取图片
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)

            # Padding: 确保长宽是 64 的倍数
            h, w = x.size(2), x.size(3)
            p_h = (64 - (h % 64)) % 64
            p_w = (64 - (w % 64)) % 64
            if p_h != 0 or p_w != 0:
                x = F.pad(x, (0, p_w, 0, p_h), mode='reflect')

            start_time = time.time()
            out_net = model(x)
            elapsed = time.time() - start_time

            # 裁剪回原来的尺寸
            x_hat = out_net['x_hat']
            x_hat = x_hat[:, :, :h, :w]
            x = x[:, :, :h, :w] # 裁剪原图以便对比

            # 限制值范围
            x_hat.clamp_(0, 1)
            
            psnr = compute_psnr(x, x_hat)
            bpp = compute_bpp(out_net)
            
            total_psnr += psnr
            total_bpp += bpp
            count += 1
            
            print(f"🖼️  {os.path.basename(img_path)} | Bpp: {bpp:.4f} | PSNR: {psnr:.2f} dB | ⏱️  {elapsed:.3f}s")

    # 打印平均结果
    if count > 0:
        print("=" * 40)
        print(f"📊 平均结果 ({count} 张图片):")
        print(f"   平均 Bpp:  {total_bpp / count:.4f}")
        print(f"   平均 PSNR: {total_psnr / count:.2f} dB")
        print("=" * 40)

if __name__ == "__main__":
    main()