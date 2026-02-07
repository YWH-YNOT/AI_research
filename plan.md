TCM (CVPR 2023) 复现与改进实战手册
目标： 在 48 小时内完成环境配置、数据准备、Baseline 复现，并开始第一轮训练。 当前状态： 服务器已就位 (RTX 3090)，待下载数据。

📅 第一阶段：总大纲与时间表
环境搭建 (0.5小时): 安装 PyTorch, CompressAI, Timm 等依赖。

数据攻坚 (3-5小时): 下载并处理 Vimeo-90k 和 Kodak 数据集（最耗时，需使用后台下载技巧）。

Baseline 验证 (0.5小时): 使用官方权重跑通测试，确立基准线。

训练启动 (24+小时): 修改代码，开始跑第一个 epoch。

🛠 第二阶段：详细执行步骤 (Step-by-Step)
步骤 1：基础环境与代码准备
在终端中依次执行以下命令。建议使用 tmux 或 screen 开启一个会话，防止网络断开导致任务中断。

Bash
# 1. 开启一个持久化会话 (名字叫 compression)
tmux new -s compression

# 2. 克隆代码仓库
git clone https://github.com/jmliu206/LIC_TCM.git
cd LIC_TCM

# 3. 安装核心依赖
# 假设你使用的是 PyTorch 镜像，只需补充以下库
pip install compressai
pip install timm
pip install scipy matplotlib pandas
pip install ninja  # 可选，加速编译
步骤 2：下载并处理数据集 (最关键一步)
Vimeo-90k 我们只需要 Septuplet (七帧序列) 版本，不要去下那个几百G的 Raw Video。

2.1 创建数据目录
Bash
# 建议放在 autodl-tmp 下，因为那是高性能盘，读取快
mkdir -p /root/autodl-tmp/data/vimeo90k
mkdir -p /root/autodl-tmp/data/kodak
2.2 下载 Vimeo-90k (Septuplet)
由于官方链接很慢，我们使用 wget 断点续传。 注意：如果服务器下载速度太慢（<1MB/s），建议在本地电脑下载好，通过百度网盘/阿里网盘传到服务器，或者使用各个算力平台的“网盘传输助手”。

尝试直接下载命令：

Bash
cd /root/autodl-tmp/data/vimeo90k

# 这是 Vimeo-90k 的官方下载链接 (Septuplet dataset)
# -c 表示断点续传，-O 指定文件名
wget -c http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip -O vimeo_septuplet.zip
解压 (这是个漫长的过程，约 82GB)：

Bash
# -q 表示安静模式，不输出一堆文件名，速度更快
unzip -q vimeo_septuplet.zip
最终文件夹结构检查： 你需要确保解压后的路径是这样的（这决定了代码能不能跑起来）：

Plaintext
/root/autodl-tmp/data/vimeo90k/vimeo_septuplet/
└── sequences/
    ├── 00001/
    ├── 00002/
    ...
2.3 下载 Kodak 测试集
Bash
cd /root/autodl-tmp/data/kodak
wget http://r0k.us/graphics/kodak/kodak/kodim01.png
# ... (一张张下太慢，可以用下面的循环脚本)
for i in {01..24}; do wget "http://r0k.us/graphics/kodak/kodak/kodim$i.png"; done
步骤 3：Baseline 快速验证 (验证环境)
在等待大数据集下载的同时，我们可以先验证代码。

下载一个预训练模型： 去 TCM GitHub 的 README 找到 Model Zoo，下载 TCM-S (MSE, lambda=0.0130) 的权重文件。 如果不方便翻墙，可以用本地电脑下好传上去。

运行评估脚本：

Bash
cd /root/LIC_TCM
# 假设权重文件名为 tcm_s_mse_0.0130.pth.tar，放在 checkpoints 文件夹
python -m compressai.utils.eval_model checkpoint checkpoints/tcm_s_mse_0.0130.pth.tar -a tcm_s -d /root/autodl-tmp/data/kodak/
预期输出： 如果你看到终端输出了 PSNR: 32.xx dB 和 Bit-rate: ...，恭喜你，环境复现成功！

步骤 4：配置训练脚本 (Train.py)
在数据下载解压完毕后，你需要修改训练指令。TCM 的代码需要一个 train.py 的启动命令。

创建启动脚本 run_train.sh： 使用 vim run_train.sh 创建文件，填入以下内容：

Bash
#!/bin/bash

# 定义数据集路径
DATASET="/root/autodl-tmp/data/vimeo90k/vimeo_septuplet"

# 启动训练
# -m: 模型名称 (tcm_s 是小模型，适合快速实验)
# --lambda: 率失真参数，建议先跑 0.0130
# --batch-size: 3090/4090 24G显存建议设为 8 或 16
# --patch-size: 必须是 256 256
# --epochs: 实验性质先跑 100 轮
# --save: 保存检查点

python train.py -m tcm_s \
  -d "$DATASET" \
  --epochs 100 \
  --lr 1e-4 \
  --batch-size 8 \
  --cuda \
  --save \
  --lambda 0.0130 \
  --checkpoint-dir "checkpoints/my_reproduce_0.0130"
给脚本权限并运行：

Bash
chmod +x run_train.sh
./run_train.sh
💡 第三阶段：避坑指南 (必读)
1. 关于 "vimeo_septuplet.zip" 下载失败
如果 wget 速度只有几十 KB/s：

方案 A: 找该服务器平台的“帮助文档”，看是否有“加速下载”或“网盘挂载”功能（如 AutoDL 的 AutoPanel）。

方案 B: 在你本地电脑用迅雷下载，然后上传。虽然慢，但比服务器断连强。

2. 关于显存溢出 (OOM)
TCM 结合了 Transformer，显存占用比纯 CNN 高。

如果报错 CUDA out of memory，请将 run_train.sh 中的 --batch-size 从 8 改为 4。

如果改为 4 还不行，检查图片是否被错误地 Resize 到了大尺寸（必须是 patch-size 256x256）。

3. 防止任务中断
一定要用 tmux 或 screen！

离开电脑前，按 Ctrl+B 然后按 D (Detach)，会话会在后台运行。

回来时，输入 tmux attach -t compression 恢复界面。

否则你关掉 SSH 窗口，下载和训练都会停止，那就白费功夫了。

🚀 下一步任务 (To-Do List)
请在接下来的几小时内完成：

[ ] 成功挂载或下载 Vimeo-90k 数据集。

[ ] 成功解压并确认 sequences 文件夹存在。

[ ] 成功运行一次 eval_model，记录下 Log。

[ ] 启动 run_train.sh，确认 Loss 在第一个 Epoch 开始下降。

一旦你看到 Loss 开始下降（例如从 0.5 降到 0.4），马上告诉我，我们就可以开始讨论如何**修改模型结构（做创新点）**了！

