"""
TCM (Transformer-based Covariance Module) 模型实现
论文: TCM: Transformer-based Covariance Module for Image Compression (CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional


class TransformerCovarianceModule(nn.Module):
    """
    变换器协方差模块 (TCM)
    用于捕获空间-特征通道之间的相关性
    """
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        # Q, K, V 投影
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        # 位置编码
        self.relative_position_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1), (2 * window_size - 1), num_heads)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 生成 Q, K, V
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape 用于多头注意力
        q = q.view(B, self.num_heads, self.head_dim, -1)
        k = k.view(B, self.num_heads, self.head_dim, -1)
        v = v.view(B, self.num_heads, self.head_dim, -1)

        # 计算注意力
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k)
        attn = attn / (self.head_dim ** 0.5)

        # 应用位置偏置
        coords_h = torch.arange(H).to(x.device)
        coords_w = torch.arange(W).to(x.device)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords = coords.flatten(1)

        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += H - 1
        relative_coords[:, :, 1] += W - 1

        # 简化: 使用全局注意力而不是窗口注意力
        attn = F.softmax(attn, dim=-1)

        # 应用注意力到 V
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)

        # Reshape 回原始形状
        out = out.view(B, C, H, W)
        out = self.proj(out)

        return out + x  # 残差连接


class TCM(CompressionModel):
    """
    TCM 基础模型类
    """
    def __init__(self, N=128, M=192, num_heads=8, **kwargs):
        super().__init__(entropy_bottleneck_channels=M)

        self.N = N
        self.M = M

        # 编码器
        self.g_a = nn.Sequential(
            nn.Conv2d(3, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2),
        )

        # 超先验编码器
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )

        # 超先验解码器
        self.h_s = nn.Sequential(
            nn.Conv2d(N, N, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, M * 2, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        # 解码器
        self.g_s = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

        # TCM 模块 (可选集成位置)
        self.tcm_encoder = TransformerCovarianceModule(N, num_heads=num_heads)
        self.tcm_decoder = TransformerCovarianceModule(N, num_heads=num_heads)

        # 上下文预测
        self.context_prediction = nn.Conv2d(M, 2 * M, kernel_size=3, stride=1, padding=1)

        # 高斯条件
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        # 编码
        y = self.g_a(x)
        y = self.tcm_encoder(y)

        # 超先验
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # 超先验解码
        gaussian_params = self.h_s(z_hat)
        scales, means = gaussian_params.chunk(2, dim=1)

        # 上下文预测
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means)

        # 解码
        x_hat = self.g_s(y_hat)
        x_hat = self.tcm_decoder(x_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """从预训练权重加载模型"""
        N = state_dict["g_a.0.weight"].shape[0]
        M = state_dict["h_s.6.weight"].shape[0] // 2
        return cls(N=N, M=M)


class TCM_S(TCM):
    """TCM 小模型"""
    def __init__(self):
        super().__init__(N=128, M=192, num_heads=8)


class TCM_M(TCM):
    """TCM 中模型"""
    def __init__(self):
        super().__init__(N=192, M=256, num_heads=12)


class TCM_L(TCM):
    """TCM 大模型"""
    def __init__(self):
        super().__init__(N=256, M=320, num_heads=16)
