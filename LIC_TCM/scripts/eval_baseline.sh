#!/bin/bash
# 使用预训练权重评估 Baseline 模型
# 使用方法: bash scripts/eval_baseline.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== Baseline 模型评估 ===${NC}"

# 预训练模型下载链接
MODEL_URL="https://github.com/jmliu206/LIC_TCM/releases/download/v1.0/tcm_s_mse_0.0130.pth.tar"
CHECKPOINT_DIR="checkpoints"
CHECKPOINT_FILE="$CHECKPOINT_DIR/tcm_s_mse_0.0130.pth.tar"

# 创建检查点目录
mkdir -p "$CHECKPOINT_DIR"

# 下载预训练模型
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo -e "${YELLOW}下载预训练模型...${NC}"
    wget -O "$CHECKPOINT_FILE" "$MODEL_URL"
else
    echo -e "${GREEN}预训练模型已存在${NC}"
fi

# 数据集路径
DATASET="${AUTODL_TMP}/data/kodak"
if [ ! -d "$DATASET" ]; then
    DATASET="/root/autodl-fs/data/kodak"
fi

if [ ! -d "$DATASET" ]; then
    echo -e "${YELLOW}Kodak 数据集未找到，请先下载${NC}"
    exit 1
fi

# 运行评估
echo -e "${YELLOW}开始评估...${NC}"
python eval.py \
    -c "$CHECKPOINT_FILE" \
    -d "$DATASET" \
    --cuda \
    -o "output/baseline"

echo -e "${GREEN}=== 评估完成 ===${NC}"
