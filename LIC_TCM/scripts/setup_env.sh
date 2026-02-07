#!/bin/bash
# TCM 环境配置脚本
# 使用方法: bash scripts/setup_env.sh

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== TCM 环境配置 ===${NC}"

# 检测 Python 版本
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "Python 版本: $PYTHON_VERSION"

# 创建虚拟环境 (如果不存在)
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    python -m venv venv
fi

# 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate

# 升级 pip
echo -e "${YELLOW}升级 pip...${NC}"
pip install --upgrade pip

# 安装 PyTorch (根据 CUDA 版本)
echo -e "${YELLOW}检测 CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1)
    echo "检测到 CUDA $CUDA_VERSION"
else
    CUDA_VERSION="cpu"
    echo "未检测到 CUDA，将安装 CPU 版本"
fi

# 安装依赖
echo -e "${YELLOW}安装依赖...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}=== 环境配置完成 ===${NC}"
echo ""
echo "使用以下命令激活环境:"
echo "  source venv/bin/activate"
echo ""
echo "或者直接运行训练:"
echo "  bash run_train.sh"
