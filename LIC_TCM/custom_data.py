import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class Vimeo90kSeptuplet(Dataset):
    def __init__(self, root, split="train", transform=None):
        """
        自定义的 Vimeo90k 数据加载器
        会自动扫描 root 目录下的所有包含 im1.png 到 im7.png 的文件夹
        """
        self.root = Path(root)
        self.transform = transform
        
        # 尝试定位到 sequences 文件夹
        if (self.root / "sequences").exists():
            self.base_dir = self.root / "sequences"
        else:
            self.base_dir = self.root

        # 扫描所有子目录，找到包含图片的文件夹
        self.samples = []
        print(f"正在扫描数据: {self.base_dir} ...")
        
        # 遍历目录寻找有效的序列文件夹 (包含 im1.png 到 im7.png)
        for root_dir, dirs, files in os.walk(self.base_dir):
            if "im1.png" in files and "im7.png" in files:
                self.samples.append(Path(root_dir))
        
        # 排序以保证训练顺序一致
        self.samples.sort()
        print(f"成功加载 {len(self.samples)} 个视频序列！")

    def __getitem__(self, index):
        seq_dir = self.samples[index]
        images = []
        # 读取 im1.png 到 im7.png
        for i in range(1, 8):
            img_path = seq_dir / f"im{i}.png"
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
        return images

    def __len__(self):
        return len(self.samples)