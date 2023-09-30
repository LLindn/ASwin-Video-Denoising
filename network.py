import math
from functools import reduce
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from acsconv.operators import ACSConv
from einops import rearrange


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size
    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(
        B,
        D // window_size[0],
        window_size[0],
        H // window_size[1],
        window_size[1],
        W // window_size[2],
        window_size[2],
        C,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        .contiguous()
        .view(-1, reduce(mul, window_size), C)
    )
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
    x = windows.view(
        B,
        D // window_size[0],
        H // window_size[1],
        W // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
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


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class WindowAttention3D(nn.Module):
    """ Window based multi-head aggregated attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = qk_scale or self.head_dim ** -0.5

        self.mlp = Mlp(in_features=dim, hidden_features=dim, out_features=dim)

        self.proj_attn = nn.Linear(2 * dim, dim)
        self.norm_attn = nn.LayerNorm(dim)

        self.register_buffer(
            "position_bias", self.get_sine_position_encoding(window_size[1:], dim // 2)
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        B_, N, C = x.shape

        attention = torch.zeros_like(x).view(
            -1, x.size(0), self.window_size[1] * self.window_size[2], self.dim
        )
        x_tmp = x.view(
            -1, x.size(0), self.window_size[1] * self.window_size[2], self.dim
        )

        qkv = (
            self.qkv(x + self.position_bias.repeat(1, self.window_size[0], 1))
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            torch.chunk(qkv[0], self.window_size[0], dim=2),
            torch.chunk(qkv[1], self.window_size[0], dim=2),
            torch.chunk(qkv[2], self.window_size[0], dim=2),
        )  # B_, nH, N/2, C

        for i in range(self.window_size[0]):
            for j in range(self.window_size[0]):
                x_att = torch.cat(
                    (
                        self.attention(
                            q[i], k[j], v[j], (B_, N // self.window_size[0], C)
                        ),
                        x_tmp[i],
                    ),
                    dim=-1,
                )
                x_att = self.norm_attn(self.proj_attn(x_att))
                attention[i] += self.mlp(x_att)

        x_out = x + (
            attention.permute(1, 0, 2, 3).contiguous().view(x.size())
            / self.window_size[0]
        )

        return torch.cat((x, x_out), dim=-1)

    def attention(self, q, k, v, x_shape):

        B_, N, C = x_shape

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.softmax(attn)

        return (attn @ v).transpose(1, 2).reshape(B_, N, C)

    def get_sine_position_encoding(self, HW, num_pos_feats=64, temperature=10000):
        """ Get sine position encoding """

        not_mask = torch.ones([1, HW[0], HW[1]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        scale = 2 * math.pi

        y_embed = y_embed / (y_embed[:, -1:, :] + 1e-6) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + 1e-6) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (
            2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_pos_feats
        )

        # BxCxHxW
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_embed = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed.flatten(2).permute(0, 2, 1).contiguous()


class ASwinBlock(nn.Module):
    """ ASwin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Spatial Window Size.
        window_size_t (int): Temporal Window size.
        shift_size (tuple[int]): Shift size for attention window.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=8,
        window_size_t=2,
        shift_size=(0, 0, 0),
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size_3D = (window_size_t, window_size, window_size)
        self.shift_size = shift_size
        self.norm2 = norm_layer(2 * dim)

        assert (
            0 <= self.shift_size[0] < self.window_size_3D[0]
        ), "shift_size must be in 0-window_size"
        assert (
            0 <= self.shift_size[1] < self.window_size_3D[1]
        ), "shift_size must be in 0-window_size"
        assert (
            0 <= self.shift_size[2] < self.window_size_3D[2]
        ), "shift_size must be in 0-window_size"

        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size_3D,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
        )
        self.mlp = Mlp(in_features=2 * dim, hidden_features=4 * dim, out_features=dim)

    def calc_attn(self, x):
        """ Forward function for aggregated shifted window attention.
        Args:
            x: Input feature, tensor size (B, D, H, W, C).
        """
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size(
            (D, H, W), self.window_size_3D, self.shift_size
        )

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(
                x,
                shifts=(-shift_size[0], -shift_size[1], -shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(
            attn_windows, window_size, B, Dp, Hp, Wp
        )  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(
                shifted_x,
                shifts=(shift_size[0], shift_size[1], shift_size[2]),
                dims=(1, 2, 3),
            )
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()

        return x

    def forward(self, x):

        return self.mlp(self.norm2((self.calc_attn(x))))


class ASwinACSconvBlock(nn.Module):
    def __init__(
        self,
        num_frames,
        conv_dim,
        trans_dim,
        num_heads,
        window_size,
        window_size_t,
        type="W",
    ):
        """ ASwinTransformer and ACSConv Block

        Args:
            num_frames (int): Number of frames.
            conv_dim (int): Feature dimension of ACSconv block.
            trans_dim (int): Feature dimension of ASwin block.
            num_heads (int): Number of attention heads.
            window_size (int): Spatial Window Size.
            window_size_t (int): Temporal Window size.
            type (str): Either 'W' for window or 'SW' for shifted window.

    """

        super(ASwinACSconvBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.window_size_t = window_size_t

        self.type = type
        self.frames = num_frames

        assert self.type in ["W", "SW"]

        if self.frames <= window_size_t:
            window_size_t = 0

        self.shift_size = (
            (0, 0, 0)
            if (self.type == "W")
            else tuple(i // 2 for i in (window_size_t, window_size, window_size))
        )

        self.trans_block = ASwinBlock(
            self.trans_dim,
            self.num_heads,
            self.window_size,
            self.window_size_t,
            self.shift_size,
        )

        self.conv1_1 = nn.Conv3d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            bias=True,
        )
        self.conv1_2 = nn.Conv3d(
            self.conv_dim + self.trans_dim,
            self.conv_dim + self.trans_dim,
            (1, 1, 1),
            (1, 1, 1),
            (0, 0, 0),
            bias=True,
        )

        self.conv_block = nn.Sequential(
            ACSConv(
                self.conv_dim,
                self.conv_dim,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                bias=False,
            ),
            nn.ReLU(True),
            ACSConv(
                self.conv_dim,
                self.conv_dim,
                (3, 3, 3),
                (1, 1, 1),
                (1, 1, 1),
                bias=False,
            ),
        )

    def forward(self, x):
        conv_x, trans_x = torch.split(
            self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1
        )
        conv_x = self.conv_block(conv_x) + conv_x

        trans_x = rearrange(trans_x, "b c d h w -> b d h w c")
        trans_x = self.trans_block(trans_x) + trans_x
        trans_x = rearrange(trans_x, "b d h w c -> b c d h w")

        return x + self.conv1_2(torch.cat((conv_x, trans_x), dim=1))


class Net(nn.Module):

    """Denoisinmg Network

        Args:
            num_frames (int): Number of frames.
            in_nc (int): Number of input channels.
            dim (int): initial Feature dimension.

    """

    def __init__(self, num_frames, in_nc=3, dim=64):
        super(Net, self).__init__()
        self.dim = dim
        self.window_size = [8, 8, 16, 16, 16, 8, 8]
        self.window_size_t = [3, 3, 6, 6, 6, 3, 3]
        self.num_heads = [2, 2, 4, 8, 4, 2, 2]
        self.block_depth = 3
        self.num_frames = num_frames
        self.dim = dim
        self.in_nc = in_nc

        self.m_head = [
            nn.Conv3d(self.in_nc, self.dim, (1, 3, 3), 1, (0, 1, 1), bias=False)
        ]

        self.m_down1 = [
            ASwinACSconvBlock(
                self.num_frames,
                self.dim // 2,
                self.dim // 2,
                self.num_heads[0],
                self.window_size[0],
                self.window_size_t[0],
                "W" if not i % 2 else "SW",
            )
            for i in range(self.block_depth)
        ] + [
            nn.Conv3d(
                self.dim, 2 * self.dim, (1, 2, 2), (1, 2, 2), (0, 0, 0), bias=False
            )
        ]

        self.m_down2 = [
            ASwinACSconvBlock(
                self.num_frames,
                self.dim,
                self.dim,
                self.num_heads[1],
                self.window_size[1],
                self.window_size_t[1],
                "W" if not i % 2 else "SW",
            )
            for i in range(self.block_depth)
        ] + [
            nn.Conv3d(
                2 * self.dim, 4 * self.dim, (1, 2, 2), (1, 2, 2), (0, 0, 0), bias=False
            )
        ]

        self.m_down3 = [
            ASwinACSconvBlock(
                self.num_frames,
                2 * self.dim,
                2 * dim,
                self.num_heads[2],
                self.window_size[2],
                self.window_size_t[2],
                "W" if not i % 2 else "SW",
            )
            for i in range(self.block_depth)
        ] + [
            nn.Conv3d(
                4 * self.dim, 4 * self.dim, (1, 2, 2), (1, 2, 2), (0, 0, 0), bias=False
            )
        ]

        self.m_body = [
            ASwinACSconvBlock(
                self.num_frames,
                2 * self.dim,
                2 * self.dim,
                self.num_heads[3],
                self.window_size[3],
                self.window_size_t[3],
                "W" if not i % 2 else "SW",
            )
            for i in range(self.block_depth)
        ]

        self.m_up3 = [
            nn.ConvTranspose3d(
                4 * self.dim, 4 * self.dim, (1, 2, 2), (1, 2, 2), (0, 0, 0), bias=False
            ),
        ] + [
            ASwinACSconvBlock(
                self.num_frames,
                2 * self.dim,
                2 * self.dim,
                self.num_heads[4],
                self.window_size[4],
                self.window_size_t[4],
                "W" if not i % 2 else "SW",
            )
            for i in range(self.block_depth)
        ]

        self.m_up2 = [
            nn.ConvTranspose3d(
                4 * self.dim, 2 * self.dim, (1, 2, 2), (1, 2, 2), (0, 0, 0), bias=False
            ),
        ] + [
            ASwinACSconvBlock(
                self.num_frames,
                self.dim,
                self.dim,
                self.num_heads[5],
                self.window_size[5],
                self.window_size_t[5],
                "W" if not i % 2 else "SW",
            )
            for i in range(self.block_depth)
        ]

        self.m_up1 = [
            nn.ConvTranspose3d(
                2 * self.dim, self.dim, (1, 2, 2), (1, 2, 2), (0, 0, 0), bias=False
            ),
        ] + [
            ASwinACSconvBlock(
                self.num_frames,
                self.dim // 2,
                self.dim // 2,
                self.num_heads[6],
                self.window_size[6],
                self.window_size_t[6],
                "W" if not i % 2 else "SW",
            )
            for i in range(self.block_depth)
        ]

        self.m_tail = [
            nn.Conv3d(self.dim, self.in_nc, (1, 3, 3), 1, (0, 1, 1), bias=False)
        ]

        self.m_head = nn.Sequential(*self.m_head)
        self.m_down1 = nn.Sequential(*self.m_down1)
        self.m_down2 = nn.Sequential(*self.m_down2)
        self.m_down3 = nn.Sequential(*self.m_down3)
        self.m_body = nn.Sequential(*self.m_body)
        self.m_up3 = nn.Sequential(*self.m_up3)
        self.m_up2 = nn.Sequential(*self.m_up2)
        self.m_up1 = nn.Sequential(*self.m_up1)
        self.m_tail = nn.Sequential(*self.m_tail)

    def forward(self, x0):

        n, h, w = x0.size()[-3:]

        paddingBottom = int(np.ceil(h / 8) * 8 - h)
        paddingRight = int(np.ceil(w / 8) * 8 - w)

        x0 = nn.ReflectionPad3d((0, paddingRight, 0, paddingBottom, 0, 0))(x0)

        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)

        x = x[..., :h, :w]

        return x
