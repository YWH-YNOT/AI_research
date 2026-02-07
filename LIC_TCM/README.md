# TCM (CVPR 2023) 复现项目

基于论文: **TCM: Transformer-based Covariance Module for Image Compression**

## 项目结构

```
LIC_TCM/
├── checkpoints/              # 模型检查点目录
├── logs/                     # TensorBoard 日志
├── models/                   # 模型定义
├── scripts/
│   ├── download_data.sh      # 数据集下载脚本 (Shell)
│   └── download_data.py      # 数据集下载脚本 (Python)
├── train.py                  # 训练脚本
├── eval.py                   # 评估脚本
├── run_train.sh              # 快速启动训练脚本
├── requirements.txt          # Python 依赖
└── README.md                 # 本文件
```

## 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境 (推荐)
conda create -n tcm python=3.10
conda activate tcm

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据集准备

**方案 A: 自动下载 (小数据集)**
```bash
# 下载 Kodak 测试集
python scripts/download_data.py
```

**方案 B: 使用下载脚本 (完整数据集)**
```bash
bash scripts/download_data.sh
```

**方案 C: 手动下载**
- Vimeo-90k Septuplet: http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
- Kodak: http://r0k.us/graphics/kodak/

### 3. 快速启动训练

```bash
bash run_train.sh
```

### 4. 查看训练进度

```bash
tensorboard --logdir logs
```

## 配置说明

### 显存配置

| GPU 显存 | 推荐批次大小 |
|---------|------------|
| 24GB (3090/4090) | 8-16 |
| 16GB (4080) | 4-8 |
| 12GB (4070) | 2-4 |

### Lambda 参数选择

| Lambda | 质量等级 | 适用场景 |
|--------|---------|---------|
| 0.0030 | 高质量 | 专业摄影 |
| 0.0130 | 中等质量 | 通用场景 (推荐) |
| 0.0480 | 低质量 | 极限压缩 |

## 常见问题

### Q: 下载速度太慢怎么办？
A: 可以在本地下载后通过网盘传输到服务器。

### Q: CUDA out of memory?
A: 在 `run_train.sh` 中将 `BATCH_SIZE` 改为 4 或更小。

### Q: 如何使用 tmux 防止断线？
A:
```bash
tmux new -s tcm_train
bash run_train.sh
# 按 Ctrl+B 然后 D 分离会话
# 回来时用: tmux attach -t tcm_train
```

## 预期结果

训练 100 epochs 后，在 Kodak 测试集上应达到:
- PSNR: ~32 dB
- MS-SSIM: ~0.92
- BPP: ~0.7

## 引用

```bibtex
@inproceedings{liu2023tcm,
  title={TCM: Transformer-based Covariance Module for Image Compression},
  author={Liu, Jianming and others},
  booktitle={CVPR},
  year={2023}
}
```

## License

本项目仅供学术研究使用。
