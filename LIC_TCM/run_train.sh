#!/bin/bash
# TCM 训练启动脚本
# 使用方法: bash run_train.sh

set -e

# ===========================
# 配置区域 (根据实际情况修改)
# ===========================

# 数据集路径
DATASET="${AUTODL_TMP}/data/vimeo90k/vimeo_septuplet"
# DATASET="/root/autodl-fs/data/vimeo90k/vimeo_septuplet"  # AutoDL 文件存储

# 模型配置
MODEL="tcm_s"          # 模型大小: tcm_s, tcm_m, tcm_l
LAMBDA=0.0130          # 率失真参数: 0.0130 (低), 0.0067 (中), 0.0030 (高)

# 训练参数
EPOCHS=100
BATCH_SIZE=8           # RTX 3090 24GB 建议值: 8 或 16
LR=1e-4

# 显存不足时改为 4
# BATCH_SIZE=4

# 保存路径
CHECKPOINT_DIR="checkpoints/my_reproduce_${LAMBDA}"
LOG_DIR="logs"

# ===========================
# 颜色输出
# ===========================
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== TCM 训练启动脚本 ===${NC}"
echo ""
echo "配置信息:"
echo "  模型:        ${MODEL}"
echo "  Lambda:      ${LAMBDA}"
echo "  Epochs:      ${EPOCHS}"
echo "  Batch Size:  ${BATCH_SIZE}"
echo "  数据集:      ${DATASET}"
echo "  检查点目录:  ${CHECKPOINT_DIR}"
echo ""

# 检查数据集是否存在
if [ ! -d "$DATASET" ]; then
    echo -e "${RED}错误: 数据集目录不存在: ${DATASET}${NC}"
    echo "请先运行数据下载脚本或检查路径设置"
    exit 1
fi

# 创建检查点目录
mkdir -p "$CHECKPOINT_DIR"

# 检测 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU 信息:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# ===========================
# 启动训练
# ===========================
echo -e "${YELLOW}开始训练...${NC}"
echo "使用 Ctrl+C 停止训练"
echo ""

python train.py \
    -m "$MODEL" \
    -d "$DATASET" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --lambda "$LAMBDA" \
    --cuda \
    --save \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --log-dir "$LOG_DIR" \
    --save-freq 10

echo ""
echo -e "${GREEN}=== 训练完成 ===${NC}"
