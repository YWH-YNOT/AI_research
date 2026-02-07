#!/bin/bash
# 训练监控脚本
# 显示 GPU 使用情况、训练日志等

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

while true; do
    clear
    echo -e "${GREEN}=== TCM 训练监控 ===${NC}"
    echo ""
    echo -e "${YELLOW}GPU 状态:${NC}"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo ""
    echo -e "${YELLOW}最近的训练日志:${NC}"
    if [ -f "logs/train.log" ]; then
        tail -n 20 logs/train.log
    else
        echo "训练日志文件不存在"
    fi
    echo ""
    echo -e "${YELLOW}按 Ctrl+C 退出${NC}"
    sleep 5
done
