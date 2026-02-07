# AI 复现实验项目

## 项目概述

本目录用于存放 AI 论文复现实验的代码和配置。

## 当前项目

### TCM (CVPR 2023)

论文标题: **TCM: Transformer-based Covariance Module for Image Compression**

**目标**: 在 48 小时内完成环境配置、数据准备、Baseline 复现，并开始训练。

## 目录结构

```
ai_experiments/
├── plan.md              # 实验计划文档
└── LIC_TCM/             # TCM 复现代码
    ├── checkpoints/     # 模型检查点
    ├── models/          # 模型定义
    ├── scripts/         # 辅助脚本
    │   ├── download_data.sh     # 数据下载
    │   ├── setup_env.sh         # 环境配置
    │   ├── eval_baseline.sh     # 评估 Baseline
    │   └── monitor_training.sh  # 训练监控
    ├── train.py         # 训练脚本
    ├── eval.py          # 评估脚本
    ├── config.py        # 配置文件
    ├── utils.py         # 工具函数
    ├── run_train.sh     # 快速启动脚本
    ├── requirements.txt # 依赖列表
    └── README.md        # 详细说明

```

## 快速开始

### 1. 克隆到服务器

```bash
git clone <your-repo-url>
cd ai_experiments/LIC_TCM
```

### 2. 配置环境

```bash
bash scripts/setup_env.sh
```

### 3. 准备数据集

```bash
# 下载 Kodak 测试集 (快速验证)
python scripts/download_data.py

# 或下载完整数据集
bash scripts/download_data.sh
```

### 4. 开始训练

```bash
# 使用 tmux 防止断线
tmux new -s tcm_train

# 启动训练
bash run_train.sh

# 分离会话: Ctrl+B 然后 D
# 恢复会话: tmux attach -t tcm_train
```

### 5. 监控训练

```bash
# 使用 TensorBoard
tensorboard --logdir logs

# 或使用监控脚本
bash scripts/monitor_training.sh
```

## 实验进度

- [x] 创建项目目录结构
- [x] 编写训练和评估脚本
- [x] 配置数据下载脚本
- [ ] 环境配置验证
- [ ] 数据集下载完成
- [ ] Baseline 验证
- [ ] 开始训练

## 参考资源

- [TCM 论文](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_TCM_Transformer-Based_Covariance_Module_for_Learned_Image_Compression_CVPR_2023_paper.html)
- [CompressAI 框架](https://github.com/InterDigitalInc/CompressAI)
- [Vimeo-90k 数据集](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip)
