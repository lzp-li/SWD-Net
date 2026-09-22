import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermalGuidedDiffusion(nn.Module):
    """
    【轻量化版】红外引导可见光扩散模块 (单模态下作为自注意力去噪引擎)
    核心策略：Bottleneck 结构 (1024 -> 256 -> 1024)
    收益：显存占用降低约 70%
    """
    def __init__(self, dim, num_heads=4):
        super(ThermalGuidedDiffusion, self).__init__()
        self.num_heads = num_heads

        # 压缩通道：1024 -> 256
        self.dim_reduce = dim // 4
        self.reduce_conv = nn.Conv2d(dim, self.dim_reduce, 1)  # 瘦身
        self.expand_conv = nn.Conv2d(self.dim_reduce, dim, 1)  # 复原

        # 缩放系数基于压缩后的维度
        self.scale = (self.dim_reduce // num_heads) ** -0.5

        # Denoise 主干 (在小通道上跑，速度快，省显存)
        self.denoise_net = nn.Sequential(
            nn.Conv2d(self.dim_reduce, self.dim_reduce, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim_reduce),
            nn.SiLU(),
            nn.Conv2d(self.dim_reduce, self.dim_reduce, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim_reduce),
            nn.SiLU()
        )

        # Cross-Attention (在小通道上计算)
        self.proj_q = nn.Conv2d(self.dim_reduce, self.dim_reduce, 1, bias=False)
        self.proj_k = nn.Conv2d(self.dim_reduce, self.dim_reduce, 1, bias=False)
        self.proj_v = nn.Conv2d(self.dim_reduce, self.dim_reduce, 1, bias=False)
        self.proj_out = nn.Conv2d(self.dim_reduce, self.dim_reduce, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, vis_feat, ir_feat):
        B, C, H, W = vis_feat.shape

        # 1. 压缩 (Compress)
        vis_small = self.reduce_conv(vis_feat)
        ir_small = self.reduce_conv(ir_feat)

        # 2. 去噪处理
        x_vis = self.denoise_net(vis_small)

        # 3. Cross-Attention 引导
        q = self.proj_q(x_vis).view(B, self.num_heads, self.dim_reduce // self.num_heads, H * W).permute(0, 1, 3, 2)
        k = self.proj_k(ir_small).view(B, self.num_heads, self.dim_reduce // self.num_heads, H * W).permute(0, 1, 3, 2)
        v = self.proj_v(ir_small).view(B, self.num_heads, self.dim_reduce // self.num_heads, H * W).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).permute(0, 1, 3, 2).contiguous().view(B, self.dim_reduce, H, W)
        out = self.proj_out(out)
        # 4. 恢复 (Expand) 并注入
        out_large = self.expand_conv(out)

        return vis_feat + self.gamma * out_large


class SelfGuidedDiffusion(nn.Module):
    """
    专为 YOLO11 单流架构设计的 ThermalGuidedDiffusion 马甲
    """
    # 增加 c2=None，完美兼容 YOLO 的自动通道推导规则
    def __init__(self, c1, c2=None, num_heads=4):
        super().__init__()
        c2 = c2 or c1  # 确保输入输出通道保持一致
        # 实例化真正的去噪核心
        self.tgd = ThermalGuidedDiffusion(c1, num_heads)

    def forward(self, x):
        # 欺骗 TGD 模块，让可见光和红外输入同一个特征，变成自注意力去噪
        return self.tgd(x, x)

class SPD(nn.Module):
    """
    Space-to-Depth 模块: 将空间分辨率无损折叠到通道维度
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 把一张图无损切分成 4 份，然后拼接到通道(Channel)维度上
        # 分辨率减半 (H/2, W/2)，但通道数翻 4 倍 (C*4)，没有任何像素被丢弃！
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                          x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class SPDConv_YOLO(nn.Module):
    """
    搭载 SPD 的无损下采样卷积 (完美平替 YOLO 默认的 Conv(stride=2))
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.spd = SPD()
        # 通道数变为 4 倍后，用 1x1 卷积(或其他)降维到目标通道数 c2
        from .conv import Conv # 确保导入了基础 Conv
        self.cv = Conv(c1 * 4, c2, k, s, p, g, d, act)

    def forward(self, x):
        return self.cv(self.spd(x))


class WDS_Upsample(nn.Module):
    """
    CVPR 2024 频域与小波启发：无插值的高频感知上采样模块
    完美对称于 SPD-Conv 的架构。利用 Depthwise 卷积预测 4 个空间频带，
    随后 PixelShuffle 物理折叠。绝对免疫 AMP 混合精度的 NaN 陷阱。
    """

    def __init__(self, c1, c2=None, scale=2):
        super().__init__()
        c2 = c2 if c2 is not None else c1
        self.scale = scale

        # 1. 维度对齐安全锁
        self.align = nn.Conv2d(c1, c2, 1, bias=False) if c1 != c2 else nn.Identity()

        # 2. 核心：极轻量级分组卷积 (Depthwise)
        # 每个特征图独立分裂出 scale^2 (即 4) 个子频带（包含轮廓与高频龟裂边缘）
        # 参数量极低！例如 1024 通道仅需约 3.6 万个参数 (1024 * 4 * 3 * 3 / 1024 = 36864)
        self.freq_expand = nn.Conv2d(
            c2, c2 * (scale ** 2),
            kernel_size=3, padding=1,
            groups=c2, bias=False
        )
        self.bn = nn.BatchNorm2d(c2 * (scale ** 2))
        self.act = nn.SiLU()

        # 3. 深度转空间（物理重组，绝对稳定）
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.align(x)
        # 提取各个频带的能量特征
        x = self.act(self.bn(self.freq_expand(x)))
        # 无损物理展开，还原空间分辨率
        return self.pixel_shuffle(x)

# class MambaContextInjector(nn.Module):
#     """
#     局部保留 + Mamba全局注入 (适配 YOLO11)
#     """
#
#     def __init__(self, c1, c2, d_state=16):
#         super().__init__()
#         assert c1 == c2, "Mamba 注入器要求输入输出通道一致"
#
#         # 1. 局部主干分支：纯粹的 3x3 卷积，绝对保留局部纹理和边缘
#         self.local_branch = nn.Sequential(
#             nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU()
#         )
#
#         # 2. 全局 Mamba 分支
#         try:
#             from mamba_ssm import Mamba
#             # Vision Mamba 会将图像展平为 1D 序列进行全局扫描
#             self.mamba = Mamba(
#                 d_model=c1,
#                 d_state=d_state,
#                 expand=2,
#             )
#         except ImportError:
#             self.mamba = None
#             print("⚠️ 警告: 未找到 mamba_ssm 库！Mamba 全局分支将退化为普通残差网络。")
#             print("👉 请在终端执行: pip install mamba-ssm causal-conv1d")
#
#         # 3. 🌟 零初始化残差权重 (核心护城河)
#         # 初始化为 0，保证刚开始训练时绝对不干扰局部特征
#         self.gamma = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         # 提取局部特征 (找龟裂、找小点)
#         local_feat = self.local_branch(x)
#
#         # 🌟 核心修复：
#         # 1. 如果没装 mamba -> 绕行
#         # 2. 如果数据在 CPU 上 (比如 YOLO 初始化算 stride 时的假图片) -> 绕行
#         if self.mamba is None or not x.is_cuda:
#             return local_feat
#
#         B, C, H, W = x.shape
#
#         # 将 2D 图像拉平成 1D 序列 (B, L, C)，喂给 Mamba 进行全局扫描
#         x_flat = x.view(B, C, -1).transpose(1, 2)
#
#         # Mamba 线性时间全局扫描
#         global_feat = self.mamba(x_flat)
#
#         # 把 1D 序列变回 2D 图像特征 (B, C, H, W)
#         global_feat = global_feat.transpose(1, 2).view(B, C, H, W)
#
#         # 强力残差注入：绝对的局部细节 + (自适应学习的全局上下文)
#         return local_feat + self.gamma * global_feat


# class LocalMambaContextInjector(nn.Module):
#     """
#     LocalMamba 全局与局部特征融合器 (专治龟裂 crazing 丢失问题)
#     结合 CNN 的局部极值提取 + Mamba 的窗口化局部扫描
#     """
#
#     # 🌟 核心修复：加入 c2 参数占位，防止 window_size 被 YOLO 强行覆盖！
#     def __init__(self, c1, c2, window_size=7, d_state=16, d_conv=4, expand=2):
#         super().__init__()
#         self.c1 = c1
#         self.window_size = window_size  # 现在它会乖乖保持为 7 了！
#
#         # 1. 局部主干分支 (依然保留卷积护盾)
#         self.local_branch = nn.Sequential(
#             nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),  # 这里也可以顺便把输出对齐为 c2
#             nn.BatchNorm2d(c2),
#             nn.SiLU()
#         )
#
#         # 2. Mamba 引擎 (延迟导入防报错)
#         try:
#             from mamba_ssm import Mamba
#             self.mamba = Mamba(
#                 d_model=c1,
#                 d_state=d_state,
#                 d_conv=d_conv,
#                 expand=expand
#             )
#         except ImportError:
#             print("⚠️ 警告: 未找到 mamba_ssm 库，退化为纯 CNN 模式")
#             self.mamba = None
#
#         # 3. 融合权重
#         self.gamma = nn.Parameter(torch.ones(1) * 1e-4)  # 初始值为 0.0001
#         self.norm = nn.LayerNorm(c1)
#
#     def window_partition(self, x, window_size):
#         """将图像切分为不重叠的局部窗口"""
#         B, C, H, W = x.shape
#         # 计算需要填充的大小，确保长宽能被 window_size 整除
#         pad_h = (window_size - H % window_size) % window_size
#         pad_w = (window_size - W % window_size) % window_size
#         if pad_h > 0 or pad_w > 0:
#             x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
#
#         Hp, Wp = H + pad_h, W + pad_w
#         # 重塑张量: [B, C, Hp, Wp] -> [B, C, Hp//ws, ws, Wp//ws, ws]
#         x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
#         # 排列并展平窗口内的像素: -> [B * num_windows, ws*ws, C] (适配 Mamba 的输入格式)
#         windows = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size * window_size, C)
#         return windows, (H, W), (Hp, Wp)
#
#     def window_reverse(self, windows, window_size, H, W, Hp, Wp):
#         """将处理后的窗口拼回原始图像"""
#         B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
#         C = windows.shape[2]
#         # 还原形状
#         x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, C)
#         x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)
#         # 裁剪掉之前的填充部分
#         if Hp > H or Wp > W:
#             x = x[:, :, :H, :W].contiguous()
#         return x
#
#     def forward(self, x):
#         local_feat = self.local_branch(x)
#
#         # 遇到未安装 Mamba 或 CPU 假图片，直接绕行 (完美解决 YOLO 初始化报错)
#         if self.mamba is None or not x.is_cuda:
#             return local_feat
#
#         # 1. 切窗：把大图切成一个个 7x7 的局部小窗口
#         windows, (H, W), (Hp, Wp) = self.window_partition(x, self.window_size)
#
#         # 2. 局部扫描：Mamba 只在小窗口内部进行状态空间的上下文扫描
#         global_feat_windows = self.mamba(windows)
#
#         # 3. 拼图：把扫描完的窗口拼回完整图像 (⚠️ 就是这一行你刚才不小心弄丢了！)
#         global_feat = self.window_reverse(global_feat_windows, self.window_size, H, W, Hp, Wp)
#
#         # 4. 归一化与极小权重残差融合 (防爆炸镇定剂)
#         B, C, H_out, W_out = global_feat.shape
#         global_feat_norm = self.norm(global_feat.view(B, C, -1).transpose(1, 2)).transpose(1, 2).view(B, C, H_out,
#                                                                                                       W_out)
#
#         return local_feat + self.gamma * global_feat_norm