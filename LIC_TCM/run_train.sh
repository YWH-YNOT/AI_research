#!/bin/bash
# TCM 训练启动脚本

set -e

# ===========================
# 配置区域
# ===========================

# 1. 确认路径 (已根据你之前的截图确认无误)
DATASET="/root/autodl-tmp/data/vimeo90k/vimeo_septuplet"

# 2. 模型配置
MODEL="tcm_s"
LAMBDA=0.0130

# 3. 显存安全配置 (针对 3080 Ti 12GB 优化)
EPOCHS=100
BATCH_SIZE=4      # <--- 已修改为 4，防止显存溢出
LR=1e-4

# 保存路径
CHECKPOINT_DIR="checkpoints/my_reproduce_${LAMBDA}"
mkdir -p "$CHECKPOINT_DIR"

# ===========================
# 启动训练
# ===========================
echo "开始训练 TCM_S (Lambda=${LAMBDA}, Batch=${BATCH_SIZE})..."

# 注意：如果下面报错 "unrecognized arguments"，请尝试去掉 --checkpoint-dir 等参数
python train.py \
    -m "$MODEL" \
    -d "$DATASET" \
    --epochs "$EPOCHS" \
    --learning-rate "$LR" \
    --batch-size "$BATCH_SIZE" \
    --lambda "$LAMBDA" \
    --cuda \
    --save \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --save-freq 10