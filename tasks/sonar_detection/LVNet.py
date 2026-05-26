import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
# from timm.models.layers import DropPath, trunc_normal_
from timm.layers import DropPath, trunc_normal_
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Conv3DMlp(nn.Module):
    """
    带有 3D 深度可分离卷积的 MLP
    打破纯 Linear 层的孤立性，强制进行局部的时空平滑，极大地增强对刚体运动目标的识别。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # 核心修改：插入 3x3x3 的深度可分离卷积 (参数量极小)
        self.dwconv3d = nn.Conv3d(
            hidden_features, hidden_features, 
            kernel_size=(3, 3, 3), padding=(1, 1, 1), groups=hidden_features
        )
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # Swin 传进来的 x 形状是 (B, D, H, W, C)
        B, D, H, W, C = x.shape
        x = self.fc1(x)
        
        # 转置为 Conv3d 需要的形状 (B, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.dwconv3d(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous() # 转回 (B, D, H, W, C)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class TemporalAxialAttention(nn.Module):
    """
    纯时序轴向注意力 (沿 D 轴)
    输入: (B, D, H, W, C) -> 重整为 (B*H*W, D, C) 计算序列自注意力
    返回: (B, D, H, W, C)
    """
    def __init__(self, dim, num_heads, qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        B, D, H, W, C = x.shape
        # 1. 将所有空间位置压入 batch 维，序列长度为 D
        x = rearrange(x, 'b d h w c -> (b h w) d c')
        N, L, C = x.shape  # N = B*H*W, L = D

        # 2. 标准的多头自注意力 (沿 D 维)
        qkv = self.qkv(x).reshape(N, L, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, N, nH, L, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]   # (N, nH, L, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (N, nH, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(N, L, C)  # (N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 3. 恢复原始形状
        x = rearrange(x, '(b h w) d c -> b d h w c', b=B, h=H, w=W)
        return x


class DecoupledSwinBlock3D(nn.Module):
    """
    解耦时空 Swin Transformer Block
    
    结构:
    - 空间支路: 2D Shifted Window Attention (帧内建模，原有逻辑)
    - 时序支路: 1D Axial Attention (帧间变化建模)
    - 两条支路串行，先空间后时序，各自配有独立的 Norm 和残差连接
    """
    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_checkpoint=False, mlp_type="mlp"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint = use_checkpoint

        # ---------- 空间注意力 (与原 Swin Block 一致) ----------
        self.norm_spatial = norm_layer(dim)
        self.spatial_attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        # ---------- 时序轴向注意力 ----------
        self.norm_temporal = norm_layer(dim)
        self.temporal_attn = TemporalAxialAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop
        )

        # ---------- 共用 MLP ----------
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp_type == "conv3d":
            from LVNet import Conv3DMlp
            self.mlp = Conv3DMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                                 act_layer=act_layer, drop=drop)
        else:
            from LVNet import Mlp
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)

    def forward_spatial(self, x, mask_matrix):
        """
        空间注意力前向过程 (与原 SwinTransformerBlock3D.forward_part1 完全一致)
        """
        B, D, H, W, C = x.shape
        x = self.norm_spatial(x)
        # pad 与 cyclic shift 逻辑完全继承原有代码
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.spatial_attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, *window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_temporal(self, x):
        """时序轴向注意力"""
        shortcut = x
        x = self.norm_temporal(x)
        x = self.temporal_attn(x)
        return shortcut + self.drop_path(x)

    def forward(self, x, mask_matrix):
        # 输入 x: (B, D, H, W, C)
        shortcut = x

        # 1. 空间支路 (Shifted Window Attention)
        x = self.forward_spatial(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        # 2. 时序支路 (Axial Attention)
        x = self.forward_temporal(x)

        # 3. MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """ Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(2,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_checkpoint=False, mlp_type="mlp"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint=use_checkpoint

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp_type == "conv3d":
            self.mlp = Conv3DMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size+(C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x, mask_matrix):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, :]  # B D H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2, :]  # B D H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2, :]  # B D H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2, :]  # B D H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B D H/2 W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


# cache each stage results
@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0],None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1],None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2],None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=(1,7,7),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 mlp_type="mlp",
                 block_type="swin"):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        BlockClass = DecoupledSwinBlock3D if block_type == "decoupled" else SwinTransformerBlock3D
        self.blocks = nn.ModuleList([
            BlockClass(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0,0,0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_type=mlp_type,
            )
            for i in range(depth)])
        
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D,H,W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(B, D, H, W, -1)
        y = x
        
        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, 'b d h w c -> b c d h w')
        return x, rearrange(y, 'b d h w c -> b c d h w')



class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, dim_scale*dim, bias=False)
        self.norm = norm_layer(dim//2)

    def forward(self, x):
        B, D, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b d h w (p1 p2 c)-> b d (h p1) (w p2) c', p1=2, p2=2, c=x.shape[-1]//4)
        x = self.norm(x)
        return x



class CNNEncoder(nn.Module):
    def __init__(self, inChans=1, outChans=12):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.conv1 = nn.Conv2d(inChans,outChans,1)
        self.conv2 = nn.Conv2d(inChans,outChans,3,padding=1)
        self.conv3 = nn.Conv2d(inChans,outChans,5,padding=2)
        self.conv4 = nn.Conv2d(inChans,outChans,7,padding=3)
        self.conv5 = nn.Conv2d(outChans*3,outChans,1)
    
    def forward(self, x):
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x3 = self.conv4(x)
        x4 = torch.cat((x1,x2,x3),dim=1)
        x5 = x if self.inChans == self.outChans else self.conv1(x)
        x = x5 + self.conv5(x4)
        return x
class CNNEncoderL(nn.Module):
    def __init__(self, inChans=1, outChans=12):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.conv1 = nn.Conv2d(inChans,outChans,1)
        self.conv2 = nn.Conv2d(inChans,outChans,3,padding=1)
        self.conv3 = nn.Conv2d(inChans,outChans,5,padding=2)
        self.conv4 = nn.Conv2d(inChans,outChans,7,padding=3)
        self.conv5 = nn.Conv2d(inChans,outChans,9,padding=4)
        self.conv6 = nn.Conv2d(outChans*4,outChans,1)
    
    def forward(self, x):
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x3 = self.conv4(x)
        x4 = self.conv5(x)
        x5 = torch.cat((x1,x2,x3,x4),dim=1)
        x6 = x if self.inChans == self.outChans else self.conv1(x)
        x = x6 + self.conv6(x5)
        return x
class CNNEncoder_Dilated(nn.Module):
    """
    针对百倍尺度差异优化的多尺度空洞编码器
    """
    def __init__(self, inChans=1, outChans=12):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        
        # 保持局部细节 (感受野 1x1)
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=1)
        # 小目标捕捉 (感受野 3x3)
        self.conv2 = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1, dilation=1)
        # 中等目标/局部上下文 (感受野 7x7)
        self.conv3 = nn.Conv2d(inChans, outChans, kernel_size=3, padding=2, dilation=2)
        # 大目标/全局上下文 (感受野 15x15)
        self.conv4 = nn.Conv2d(inChans, outChans, kernel_size=3, padding=4, dilation=4)
        
        self.conv5 = nn.Conv2d(outChans * 3, outChans, kernel_size=1)
        
        # 引入一个通道注意力(SE block)来动态选择不同尺度的重要性
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(outChans, outChans // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outChans // 2, outChans, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x3 = self.conv4(x)
        
        x4 = torch.cat((x1, x2, x3), dim=1)
        x_fuse = self.conv5(x4)
        
        # 尺度自适应加权
        x_fuse = x_fuse * self.se(x_fuse)
        
        x_res = x if self.inChans == self.outChans else self.conv1(x)
        return x_res + x_fuse
class SPPModule(nn.Module):
    """
    空间金字塔池化模块 (Spatial Pyramid Pooling)
    专为极度尺度差异和密集梯度优化设计
    """
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 3, 5)):
        super().__init__()
        self.pool_sizes = pool_sizes
        
        # 构建不同尺度的自适应平均池化层
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in pool_sizes
        ])
        
        # 融合层：原本的通道数 + 所有池化尺度产生的通道数
        # 比如 pool_sizes有3个，那就拼接了 1(原图) + 3(池化) = 4 份特征
        fuse_in_channels = in_channels * (len(pool_sizes) + 1)
        
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(fuse_in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 列表第一项保留高频的原始局部特征
        features = [x]
        
        # 生成不同宏观尺度的特征并上采样回原图大小
        for pool in self.pools:
            # 1. 池化压缩 (提取宏观上下文)
            p = pool(x)
            # 2. 上采样放大 (将宏观语义广播回每一个像素)
            p = F.interpolate(p, size=(h, w), mode='bilinear', align_corners=False)
            features.append(p)
            
        # 在通道维度拼接所有特征
        out = torch.cat(features, dim=1)
        
        # 1x1 卷积融合特征并降维
        out = self.fuse_conv(out)
        
        return out
class CNNEncoder_with_SPP(nn.Module):
    def __init__(self, inChans=1, outChans=12):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        
        # 1. 提取基础局部特征 (保留 3x3 捕捉微小目标的能力)
        self.base_conv = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1)
        
        # 2. 接入 SPP 模块获取百倍感受野 (分别获取 1x1, 3x3, 5x5 的全局特征)
        self.spp = SPPModule(in_channels=outChans, out_channels=outChans, pool_sizes=(1, 3, 5))
        
        self.shortcut = nn.Conv2d(inChans, outChans, 1) if inChans != outChans else nn.Identity()
    
    def forward(self, x):
        res = self.shortcut(x)
        
        # 提取局部细节
        x = self.base_conv(x)
        
        # 提取宏观多尺度上下文
        x = self.spp(x)
        
        return res + x
class CNNEncoder_LK(nn.Module):
    """
    大核深度可分离卷积编码器 (Large Kernel DW-Conv)
    完美兼容 Muon-Adam (梯度平滑无网格效应) 且 速度极快 (无插值、极低参数量)
    """
    def __init__(self, inChans=1, outChans=12):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        
        # 旁支 1: 提取局部高频细节 (保留标准的 3x3 卷积)
        self.conv1 = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1)
        
        # 旁支 2: 中等宏观视野 (使用 7x7 深度可分离卷积)
        self.conv2 = nn.Sequential(
            # DW 层：负责大范围空间信息提取 (groups=inChans 是降低参数和加速的核心)
            nn.Conv2d(inChans, inChans, kernel_size=7, padding=3, groups=inChans),
            # PW 层：负责跨通道信息融合
            nn.Conv2d(inChans, outChans, kernel_size=1)
        )
        
        # 旁支 3: 超大宏观视野 (使用 11x11 深度可分离卷积，替代 ASPP 的作用)
        self.conv3 = nn.Sequential(
            nn.Conv2d(inChans, inChans, kernel_size=11, padding=5, groups=inChans),
            nn.Conv2d(inChans, outChans, kernel_size=1)
        )
        
        # 将 3 个尺度的特征融合
        self.fuse = nn.Conv2d(outChans * 3, outChans, kernel_size=1)
        self.shortcut = nn.Conv2d(inChans, outChans, 1) if inChans != outChans else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        # 在通道维度拼接
        x_cat = torch.cat((x1, x2, x3), dim=1)
        
        # 融合并加上残差
        return res + self.fuse(x_cat)
class CNNEncoder_AvgSPPF(nn.Module):
    """
    基于平滑梯度的快速空间金字塔池化编码器 (Avg-SPPF)
    速度极快，无插值操作，梯度平滑完美兼容 Adam，且能稀释高亮旁瓣
    """
    def __init__(self, inChans=1, outChans=12):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        
        # 1. 基础特征提取 (保留局部高频信息)
        self.conv_base = nn.Conv2d(inChans, outChans, kernel_size=3, padding=1)
        
        # 2. 降维卷积，减少后续 SPPF 拼接时的计算和显存压力
        self.conv_reduce = nn.Conv2d(outChans, outChans // 2, kernel_size=1)
        
        # 3. 核心：AvgPool，stride=1 保证分辨率不降，免去插值
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2, count_include_pad=False)
        
        # 4. 融合层：1 份原始输入 + 3 份池化特征 = 4 份特征
        self.conv_fuse = nn.Conv2d((outChans // 2) * 4, outChans, kernel_size=1)
        
        self.shortcut = nn.Conv2d(inChans, outChans, 1) if inChans != outChans else nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        
        # 局部特征提取
        x_base = self.conv_base(x)
        
        # 降维进入 SPPF 核心区
        x_reduced = self.conv_reduce(x_base)
        
        # 串行套娃平均池化 (等效于感受野 5, 9, 13)
        y1 = self.avg_pool(x_reduced)
        y2 = self.avg_pool(y1)
        y3 = self.avg_pool(y2)
        
        # 拼接并在通道维度融合
        x_sppf = torch.cat([x_reduced, y1, y2, y3], dim=1)
        x_fuse = self.conv_fuse(x_sppf)
        
        return res + x_fuse


class MotionCNNEncoder(nn.Module):
    """
    带有显式帧间差分的运动感知编码器
    核心逻辑：静态旁瓣相减为 0 被消除，运动目标相减产生强烈高频激活，直接在浅层秒杀旁瓣！
    """
    def __init__(self, inChans=1, outChans=12, num_frame=4):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.num_frame = num_frame
        
        # 1. 保留你原始证明有效的空间多尺度特征
        self.conv1 = nn.Conv2d(inChans, outChans, 1)
        self.conv2 = nn.Conv2d(inChans, outChans, 3, padding=1)
        self.conv3 = nn.Conv2d(inChans, outChans, 5, padding=2)
        self.conv4 = nn.Conv2d(inChans, outChans, 7, padding=3)
        
        # 2. 【新增】专门处理时间差分（运动信息）的卷积层
        self.diff_conv = nn.Conv2d(inChans, outChans, 3, padding=1)
        
        # 3. 融合层：3路空间特征 + 1路运动特征 = 4倍 outChans
        self.conv5 = nn.Conv2d(outChans * 4, outChans, 1)
    
    def forward(self, x_flat):
        # x_flat 的形状是 (B * D, C, H, W)
        bd, c, h, w = x_flat.shape
        d = self.num_frame
        b = bd // d
        
        # --- 空间特征提取 ---
        x1 = self.conv2(x_flat)
        x2 = self.conv3(x_flat)
        x3 = self.conv4(x_flat)
        
        # --- 【核心】提取时间运动特征 ---
        # 还原回 5D 时空张量 (B, D, C, H, W)
        x_3d = x_flat.view(b, d, c, h, w)
        diff = torch.zeros_like(x_3d)
        
        # 显式计算相邻帧的差异 (当前帧 - 前一帧)
        # 静态旁瓣相减后接近 0，运动目标会留下极强的高低电平反差
        diff[:, 1:] = x_3d[:, 1:] - x_3d[:, :-1]
        diff[:, 0] = x_3d[:, 1] - x_3d[:, 0]  # 第一帧的 fallback
        
        # 压平回去并提取运动特征
        diff_flat = diff.view(bd, c, h, w)
        x_diff = self.diff_conv(diff_flat)
        
        # --- 融合时空信息 ---
        # 将空间形状 (x1, x2, x3) 和 纯运动信息 (x_diff) 拼接
        x4 = torch.cat((x1, x2, x3, x_diff), dim=1)
        
        x5 = x_flat if self.inChans == self.outChans else self.conv1(x_flat)
        x_out = x5 + self.conv5(x4)
        
        return x_out
class STSF(nn.Module):
    """
    基于时间方差 (Temporal Variance) 的运动感知编码器
    核心逻辑：静态/缓慢变化的高亮旁瓣方差为0，运动目标的轨迹方差极大。
    用统计学特征彻底粉碎基于亮度的误检！
    """
    def __init__(self, inChans=1, outChans=12, num_frame=4):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.num_frame = num_frame
        
        # 1. 保留原本提取空间形状特征的卷积 (用于勾勒目标的精确轮廓)
        self.conv1 = nn.Conv2d(inChans, outChans, 1)
        self.conv2 = nn.Conv2d(inChans, outChans, 3, padding=1)
        self.conv3 = nn.Conv2d(inChans, outChans, 5, padding=2)
        self.conv4 = nn.Conv2d(inChans, outChans, 7, padding=3)
        
        # 2. 【新增】专门处理“时间方差热力图”的卷积层
        self.var_conv = nn.Conv2d(inChans, outChans, 3, padding=1)
        
        # 3. 融合层：3路空间形状 + 1路运动能量 = 4倍 outChans
        self.conv5 = nn.Conv2d(outChans * 4, outChans, 1)
    
    def forward(self, x_flat):
        # x_flat 形状: (B * D, C, H, W)
        bd, c, h, w = x_flat.shape
        d = self.num_frame
        b = bd // d
        
        # --- 提取静态空间特征 (会同时提取到目标和旁瓣) ---
        x1 = self.conv2(x_flat)
        x2 = self.conv3(x_flat)
        x3 = self.conv4(x_flat)
        
        # --- 【核心：用户提出的时间方差特征】 ---
        # 1. 还原为 5D 时空张量 (B, D, C, H, W)
        x_3d = x_flat.view(b, d, c, h, w)
        
        # 2. 沿着时间维度 D (dim=1) 计算像素级方差
        # unbiased=False 防止帧数太少时引发除零或 NaN 异常
        # keepdim=True 保持形状为 (B, 1, C, H, W)
        temporal_var = torch.var(x_3d, dim=1, keepdim=True, unbiased=False)
        
        # 3. 将这张浓缩了 4 帧运动能量的“热力图”，复制/广播回 4 帧的维度
        # 这样每一帧在做决策时，都能“看”到整个短视频的运动热力分布
        temporal_var_expanded = temporal_var.expand(-1, d, -1, -1, -1)
        
        # 4. 压平回 4D 并进行特征映射
        var_flat = temporal_var_expanded.reshape(bd, c, h, w)
        x_var = self.var_conv(var_flat)
        
        # --- 特征融合 ---
        # 网络会自动学习到：只有当 x1, x2, x3 (空间上有物体) 
        # 且 x_var (时间上有剧烈运动) 同时存在时，才输出高响应！
        x4 = torch.cat((x1, x2, x3, x_var), dim=1)
        
        x5 = x_flat if self.inChans == self.outChans else self.conv1(x_flat)
        x_out = x5 + self.conv5(x4)
        
        return x_out
class CoordAtt(nn.Module):
    """
    坐标注意力机制 (Coordinate Attention)
    赋予网络极强的“X-Y 轴几何形状”感知能力
    """
    def __init__(self, inp, reduction=8):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        # 必须针对 2D 特征图操作，此时 x 形状 (BD, C, H, W)
        n, c, h, w = x.size()
        
        # 沿高度和宽度独立提取形状投影
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成基于 X-Y 几何形状的注意力掩码
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        # 屏蔽掉不符合目标形状特征的区域
        return identity * a_w * a_h
class TemporalShortcutGate(nn.Module):
    """
    3D 时间门控滤网
    在拼接浅层毒药(Shortcut)前，用 1x3x3 时序卷积计算权重，过滤掉其中静态不变的高亮旁瓣
    """
    def __init__(self, dim, num_frame=4):
        super().__init__()
        self.num_frame = num_frame
        # 时间维度感受野为3，空间为1。提取纯粹的运动强度
        self.gate = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim),
            nn.Sigmoid()
        )

    def forward(self, x_2d):
        # 接收的 x_2d 形状是 (B*D, C, H, W)
        BD, C, H, W = x_2d.shape
        B = BD // self.num_frame
        D = self.num_frame
        
        # 还原回 3D 状态 (B, C, D, H, W)
        x_3d = x_2d.view(B, D, C, H, W).permute(0, 2, 1, 3, 4)
        
        # 计算时序门控权重 (有明显帧间运动的区域权重高，静态旁瓣权重趋近于 0)
        weight = self.gate(x_3d)
        x_3d_filtered = x_3d * weight
        
        # 重新拍扁回 2D 交给解码器
        return x_3d_filtered.permute(0, 2, 1, 3, 4).reshape(BD, C, H, W)
class CNNDownSample(nn.Module):
    def __init__(self, inChans):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.conv = nn.Conv2d(inChans*4,inChans*2,1)
    
    def forward(self,x):
        x = self.unshuffle(x)
        x = self.conv(x)
        return x
    
class CNNDecoder(nn.Module):
    def __init__(self, inChans=48, outChans=24):
        super().__init__()
        self.inChans = inChans
        self.outChans = outChans
        self.conv1 = nn.Conv2d(inChans,outChans,1)
        self.conv2 = nn.Conv2d(outChans,outChans,3,padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        return x
    
class CNNUpSample(nn.Module):
    def __init__(self, inChans):
        super().__init__()
        self.conv = nn.Conv2d(inChans,inChans*2,1)
        self.shuffle = nn.PixelShuffle(2)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.shuffle(x)
        return x
    


class LVNet(nn.Module):
    """ 
    Args:
        num_frame (int): Number of frames of the input image. Default: 4.
        embed_dim (int): Dims of the patch embedding. Default: 24.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: (4,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 num_frame=4,
                 embed_dim=24,
                 depths=[2, 2, 2, 1],
                 num_heads=[3, 6, 12, 24],
                 window_size=(4,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 encoder_type="cnn",
                 mlp_type="mlp",
                 block_type="swin"):
        super().__init__()
        EncoderClass = STSF if encoder_type == "stsf" else CNNEncoder
        if encoder_type == "stsf":
            self.cnnEncoderL1 = EncoderClass(1, embed_dim//4, num_frame=num_frame)
            self.cnnDownL1 = CNNDownSample(embed_dim//4)
            self.cnnEncoderL2 = EncoderClass(embed_dim//2, embed_dim//2, num_frame=num_frame)
            self.cnnDownL2 = CNNDownSample(embed_dim//2)
            self.cnnEncoderL3 = EncoderClass(embed_dim, embed_dim, num_frame=num_frame)
        else:
            self.cnnEncoderL1 = CNNEncoder(1,embed_dim//4)
            self.cnnDownL1 = CNNDownSample(embed_dim//4)
            self.cnnEncoderL2 = CNNEncoder(embed_dim//2,embed_dim//2)
            self.cnnDownL2 = CNNDownSample(embed_dim//2)
            self.cnnEncoderL3 = CNNEncoder(embed_dim,embed_dim)

        self.cnnUpL1 = CNNUpSample(embed_dim)
        self.cnnDecoderL1 = CNNDecoder(embed_dim,embed_dim//2)
        self.cnnUpL2 = CNNUpSample(embed_dim//2)
        self.cnnDecoderL2 = CNNDecoder(embed_dim//2,embed_dim//4)

        self.num_frame = num_frame 
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        self.layers_down = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i_layer<self.num_layers-1 else None,
                use_checkpoint=use_checkpoint,
                mlp_type=mlp_type,
                block_type=block_type)
            self.layers_down.append(layer)
            
        self.layers_up = nn.ModuleList()
        self.layers_ct = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
                int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)) if i_layer > 0 else embed_dim * 2 ** (self.num_layers-1),
                depth=depths[(self.num_layers-1-i_layer)],
                num_heads=num_heads[(self.num_layers-1-i_layer)],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                norm_layer=norm_layer,
                downsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                mlp_type=mlp_type,
                block_type=block_type)
            self.layers_up.append(layer)
            self.layers_ct.append(concat_linear)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))
        self.norm = norm_layer(self.num_features)
        self.endConv = nn.Conv2d(embed_dim//4,1,1)
        
        # L2 的输出通道是 12，L1 的输出通道是 6
        # self.shape_gate_L2 = CoordAtt(inp=embed_dim // 2)
        # self.shape_gate_L1 = CoordAtt(inp=embed_dim // 4)
        
        # =========== 新增：时间跳跃连接门控 ===========
        # cnnShortCut 弹出的 L2 通道是 embed_dim//2，L1 通道是 embed_dim//4
        # self.ts_gate_L2 = TemporalShortcutGate(embed_dim // 2, num_frame=num_frame)
        # self.ts_gate_L1 = TemporalShortcutGate(embed_dim // 4, num_frame=num_frame)
        
        import math
        pi = 0.01
        # 这一行能保证第一轮训练时，模型默认所有像素都是背景，Focal Loss 不会爆表
        nn.init.constant_(self.endConv.bias, -math.log((1 - pi) / pi))
    def forward(self, x):
        """Forward function."""
        B = x.shape[0]  # x: (4, 1, 4, 512, 512)
        x = rearrange(x, 'b c d h w -> (b d) c h w')# x: (16, 1, 512, 512)  [4*4=16, 将batch和帧数合并]
        cnnShortCut = []
        x = self.cnnEncoderL1(x)# x: (16, 6, 512, 512)  [embed_dim=24, 24//4=6]
        cnnShortCut.append(x)# 保存用于跳跃连接
        x = self.cnnDownL1(x)# PixelUnshuffle(2): (16, 6, 512, 512) -> (16, 24, 256, 256) [6*4=24]# 然后1x1卷积降维到 12 通道: (16, 12, 256, 256) [embed_dim//2 = 12]
        x = self.cnnEncoderL2(x)# x: (16, 12, 256, 256)  [embed_dim//2 = 12]
        cnnShortCut.append(x) # 保存用于跳跃连接
        x = self.cnnDownL2(x)# PixelUnshuffle(2): (16, 12, 256, 256) -> (16, 48, 128, 128) [12*4=48] # 然后1x1卷积降维到 24 通道: (16, 24, 128, 128) [embed_dim = 24]
        x = self.cnnEncoderL3(x)# x: (16, 24, 128, 128)  [embed_dim = 24]
        x = rearrange(x, '(b d) c h w -> b c d h w', b = B)# 转换回5D格式进入Swin Transformer# x: (4, 24, 4, 128, 128)  [恢复batch和帧维度]

        shortcuts = []
        for layer in self.layers_down:
            x,y = layer(x.contiguous())
            # layer 0: x: (4, 48, 4, 64, 64),   y: (4, 24, 4, 128, 128)  [有PatchMerging]
            # layer 1: x: (4, 96, 4, 32, 32),   y: (4, 48, 4, 64, 64)    [有PatchMerging]
            # layer 2: x: (4, 192, 4, 16, 16),  y: (4, 96, 4, 32, 32)    [有PatchMerging]
            # layer 3: x: (4, 192, 4, 16, 16),  y: (4, 192, 4, 16, 16)   [无PatchMerging, 最后一层]
            shortcuts.append(y)
        shortcuts.pop()# 移除最后一个，因为decoder不需要它（已经是最底层）# shortcuts现在: [y0(24), y1(48), y2(96)]，从深到浅排列
        
        x = rearrange(x, 'n c d h w -> n d h w c')# x: (4, 4, 16, 16, 192)
        x = self.norm(x)
        x = rearrange(x, 'n d h w c -> n c d h w')# x: (4, 192, 4, 16, 16)  [回到原始格式]
        
        # ====== Swin Transformer Decoder (上采样路径) ======
        x,_ = self.layers_up[0](x)# layer 0 (最深层): x: (4, 192, 4, 16, 16) -> (4, 96, 4, 32, 32)  [PatchExpand]
        
        for layer, linear in zip(self.layers_up[1:], self.layers_ct[1:]):# 第1次迭代 (i=1): 对应encoder的layer 2
            x = torch.cat([x, shortcuts.pop()], dim=1)# x: (4, 96+96=192, 4, 32, 32)  [拼接skip connection]
            x = rearrange(x, 'n c d h w -> n d h w c')# x: (4, 4, 32, 32, 192)
            x = linear(x)  # linear: 192 -> 96# x: (4, 4, 32, 32, 96)
            x = rearrange(x, 'n d h w c -> n c d h w')# x: (4, 96, 4, 32, 32)
            x,_ = layer(x)# layer 1: x: (4, 48, 4, 64, 64)  [PatchExpand]

        # ====== CNN Decoder ======
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        # x: (16, 24, 128, 128)
        
        x = self.cnnUpL1(x)
        # conv: (16, 24, 128, 128) -> (16, 48, 128, 128) [24*2=48]
        # PixelShuffle(2): (16, 48, 128, 128) -> (16, 12, 256, 256) [48/4=12]
        
        # x = torch.cat([x, x], dim=1)
        x = torch.cat([x, cnnShortCut.pop()], dim=1)
        # x = torch.cat([x, self.shape_gate_L2(cnnShortCut.pop())], dim=1)#CoordAtt 
        # x = torch.cat([x, self.ts_gate_L2(cnnShortCut.pop())], dim=1)#TemporalShortcutGate
        # x: (16, 12+12=24, 256, 256)  [拼接L2的shortcut]
        
        x = self.cnnDecoderL1(x)
        # conv1: (16, 24, 256, 256) -> (16, 12, 256, 256)
        # 然后 + conv2(12->12): 输出 (16, 12, 256, 256)
    
        x = self.cnnUpL2(x)
        # conv: (16, 12, 256, 256) -> (16, 24, 256, 256) [12*2=24]
        # PixelShuffle(2): (16, 24, 256, 256) -> (16, 6, 512, 512) [24/4=6]
        
        # x = torch.cat([x, x], dim=1)
        x = torch.cat([x, cnnShortCut.pop()], dim=1)
        # x = torch.cat([x, self.shape_gate_L1(cnnShortCut.pop())], dim=1)#CoordAtt 
        # x = torch.cat([x, self.ts_gate_L1(cnnShortCut.pop())], dim=1)#TemporalShortcutGate
        # x: (16, 6+6=12, 512, 512)  [拼接L1的shortcut]
        
        x = self.cnnDecoderL2(x)
        # conv1: (16, 12, 512, 512) -> (16, 6, 512, 512)
        # 然后 + conv2(6->6): 输出 (16, 6, 512, 512)
        
        x = self.endConv(x)
        # x: (16, 1, 512, 512)
        
        x = rearrange(x, '(b d) c h w -> b c d h w', b = B)
        # x: (4, 1, 4, 512, 512)  [最终输出，与输入形状一致]
        return x