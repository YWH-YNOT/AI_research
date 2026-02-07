#!/bin/bash
# TCM 数据集下载脚本
# 使用方法: bash scripts/download_data.sh

set -e  # 遇到错误立即退出

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== TCM 数据集下载脚本 ===${NC}"

# 数据根目录 (可根据实际情况修改)
DATA_ROOT="${AUTODL_TMP:-/root/autodl-tmp}/data"
mkdir -p "$DATA_ROOT"

# ===========================
# 1. Vimeo-90k (Septuplet) 数据集
# ===========================
echo -e "${YELLOW}正在下载 Vimeo-90k Septuplet 数据集...${NC}"

VIMEO_DIR="$DATA_ROOT/vimeo90k"
mkdir -p "$VIMEO_DIR"

cd "$VIMEO_DIR"

# 断点续传下载
if [ ! -f "vimeo_septuplet.zip" ]; then
    echo -e "${GREEN}开始下载 vimeo_septuplet.zip (约 82GB)...${NC}"
    wget -c http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip -O vimeo_septuplet.zip
else
    echo -e "${GREEN}vimeo_septuplet.zip 已存在，跳过下载${NC}"
fi

# 解压
if [ ! -d "vimeo_septuplet/sequences" ]; then
    echo -e "${GREEN}正在解压 (这需要一些时间)...${NC}"
    unzip -q vimeo_septuplet.zip
    echo -e "${GREEN}解压完成！${NC}"
else
    echo -e "${GREEN}Vimeo-90k 数据集已解压${NC}"
fi

# 验证目录结构
if [ -d "vimeo_septuplet/sequences" ]; then
    echo -e "${GREEN}✓ Vimeo-90k 数据集结构正确${NC}"
    ls -la vimeo_septuplet/sequences | head -5
else
    echo -e "${RED}✗ 目录结构不正确，请检查！${NC}"
    exit 1
fi

# ===========================
# 2. Kodak 测试集
# ===========================
echo -e "${YELLOW}正在下载 Kodak 测试集...${NC}"

KODAK_DIR="$DATA_ROOT/kodak"
mkdir -p "$KODAK_DIR"

cd "$KODAK_DIR"

# 批量下载 24 张 Kodak 图片
for i in {01..24}; do
    if [ ! -f "kodim$i.png" ]; then
        wget -q "http://r0k.us/graphics/kodak/kodak/kodim$i.png"
        echo -e "${GREEN}下载 kodim$i.png${NC}"
    fi
done

echo -e "${GREEN}✓ Kodak 测试集下载完成 (${#kodim*.png} 张图片)${NC}"

# ===========================
# 3. 可选: Tecnick 测试集
# ===========================
echo -e "${YELLOW}是否下载 Tecnick 测试集? (可选，输入 y 下载)${NC}"
read -r -p "> " answer

if [ "$answer" = "y" ]; then
    TECHNIQUE_DIR="$DATA_ROOT/technique"
    mkdir -p "$TECHNIQUE_DIR"

    cd "$TECHNIQUE_DIR"
    wget -c http://mmonshori.com/res/dataset/technique.zip
    unzip -q technique.zip
    echo -e "${GREEN}✓ Tecnick 测试集下载完成${NC}"
fi

# ===========================
# 总结
# ===========================
echo -e "${GREEN}=== 数据集下载完成 ===${NC}"
echo ""
echo "数据集位置:"
echo "  Vimeo-90k: $VIMEO_DIR/vimeo_septuplet"
echo "  Kodak:     $KODAK_DIR"
echo ""
echo -e "${YELLOW}下一步: 运行评估脚本验证环境${NC}"
echo "  python -m compressai.utils.eval_model checkpoint checkpoints/tcm_s_mse_0.0130.pth.tar -a tcm_s -d $KODAK_DIR"
