import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional

class TransformerCovarianceModule(nn.Module):
    """ 变换器协方差模块 (TCM) """
    def __init__(self, dim, num_heads=8, window_size=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        # Q, K, V 投影
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # 生成 Q, K, V
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape 用于多头注意力
        q = q.view(B, self.num_heads, self.head_dim, -1)
        k = k.view(B, self.num_heads, self.head_dim, -1)
        v = v.view(B, self.num_heads, self.head_dim, -1)

        # 计算注意力 (简化版全局注意力)
        attn = torch.einsum('bhdn,bhdm->bhnm', q, k)
        attn = attn / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # 应用注意力到 V
        out = torch.einsum('bhnm,bhdm->bhdn', attn, v)
        
        # Reshape 回原始形状
        out = out.reshape(B, C, H, W) # 修复 view 的潜在问题
        out = self.proj(out)
        return out + x  # 残差连接

class TCM(CompressionModel):
    def __init__(self, N=128, M=192, num_heads=8, **kwargs):
        super().__init__(entropy_bottleneck_channels=N) # 注意：这里通常是 N (超先验通道)

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
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2), # 输出 M
        )

        # 超先验编码器
        self.h_a = nn.Sequential(
            nn.Conv2d(M, N, kernel_size=3, stride=1, padding=1), # 输入 M
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
        )

        # 超先验解码器
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(N, M * 2, kernel_size=3, stride=1, padding=1), # 这里的最后一层通常是 Conv2d
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

        # TCM 模块 (关键修复：使用 M 而不是 N，因为它们接在 g_a 后面)
        self.tcm_encoder = TransformerCovarianceModule(M, num_heads=num_heads)
        self.tcm_decoder = TransformerCovarianceModule(M, num_heads=num_heads)

        # 高斯条件
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        # 编码
        y = self.g_a(x)
        y = self.tcm_encoder(y) # y 的形状是 [B, M, H, W]

        # 超先验
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # 超先验解码 (预测均值和方差)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, dim=1)

        # 上下文预测 + 高斯条件
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)

        # 解码
        y_hat_dec = self.tcm_decoder(y_hat) # 先过 TCM
        x_hat = self.g_s(y_hat_dec)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
        }

# 定义这三个类，train.py 会用到
class TCM_S(TCM):
    def __init__(self, **kwargs):
        super().__init__(N=128, M=192, num_heads=8, **kwargs)

class TCM_M(TCM):
    def __init__(self, **kwargs):
        super().__init__(N=192, M=256, num_heads=8, **kwargs)

class TCM_L(TCM):
    def __init__(self, **kwargs):
        super().__init__(N=256, M=320, num_heads=8, **kwargs)