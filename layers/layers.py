import torch
import torch.nn as nn
from layers.RMSNorm import RMSNorm
from layers.DropPath import DropPath
import torch.nn.functional as F
import math


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


def op(x_1, x_2):
    return torch.cat([x_1[:, :, :x_2.shape[2], :x_2.shape[3]], x_2], dim=1)


class UpSample(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.down(x)


class Block(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 groups=32,
                 drop_out=0.1
                 ):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Dropout(drop_out),
            nn.Conv2d(dim, dim_out, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)



class BasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 norm_groups=32,
                 drop_out=0.1,
                 with_attn=False,
                 n_heads=1,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.with_attn = with_attn
        self.block_1 = nn.Sequential(
            Block(dim, dim_out, groups=norm_groups, drop_out=drop_out),
        )
        self.block_2 = nn.Sequential(
            Block(dim_out, dim_out, groups=norm_groups, drop_out=drop_out),
            Block(dim_out, dim_out, groups=norm_groups, drop_out=drop_out),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        if self.with_attn:
            self.drop_path = DropPath(drop_prob=drop_path)
            self.attn = SelfAttention(dim_out, n_heads=n_heads, norm_groups=norm_groups)

    def forward(self, x):
        h = self.block_1(x)
        h = self.block_2(h) + self.res_conv(x)
        if self.with_attn:
            h = h + self.drop_path(self.attn(h))
        return h


class ResnetBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 norm_groups=32,
                 drop_out=0.1,
                 time_dim=32,
                 condition_dim=0,
                 with_attn=False,
                 n_heads=1,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.with_attn = with_attn
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out),
        )
        self.block_1 = nn.Sequential(
            Block(dim + condition_dim, dim_out, groups=norm_groups, drop_out=drop_out),
        )

        self.block_2 = nn.Sequential(
            Block(dim_out, dim_out, groups=norm_groups, drop_out=drop_out),
            Block(dim_out, dim_out, groups=norm_groups, drop_out=drop_out)
        )

        if self.with_attn:
            self.drop_path = DropPath(drop_prob=drop_path)
            self.attn = SelfAttention(dim_out, n_heads=n_heads, norm_groups=norm_groups)

    def forward(self, x, t, condition=None):
        h = x
        if condition is not None:
            h = torch.cat([h, condition], dim=1)
        h = self.block_1(h)
        t = self.time_mlp(t).view(h.shape[0], -1, 1, 1)
        h = h + t
        h = self.block_2(h) + self.res_conv(x)
        if self.with_attn:
            h = h + self.drop_path(self.attn(h))
        return h


class SelfAttention(nn.Module):
    def __init__(self,
                 in_channel,
                 scale=1,
                 n_heads=1,
                 norm_groups=32,
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = in_channel * scale // self.n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.proj_qkv = nn.Conv2d(in_channel, in_channel * scale * 3, 1, bias=False)
        self.proj_out = nn.Conv2d(in_channel * scale, in_channel, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x)
        q, k, v = self.proj_qkv(x).view(B, self.n_heads, self.head_dim * 3, H * W).permute(0, 1, 3, 2).chunk(3, -1)
        dots = q @ k.transpose(-2, -1) * self.scale
        dots = dots.softmax(dim=-1)
        h = dots @ v
        h = h.transpose(-1, -2).reshape(B, -1, H, W)
        h = self.proj_out(h)
        return h


class Attention(nn.Module):
    def __init__(self,
                 num_embeddings=2,
                 hidden_dim=32,
                 scale=1,
                 n_heads=1,
                 drop_path=0.1,
                 ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim * scale // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.norm = RMSNorm(hidden_dim)

        self.embed_q = nn.Embedding(num_embeddings, hidden_dim)
        self.proj_q = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * scale),
            nn.SiLU(),
            nn.Linear(hidden_dim * scale, hidden_dim * scale),
        )
        self.proj_kv = nn.Linear(hidden_dim, hidden_dim * scale * 2, bias=False)
        self.proj_out = nn.Linear(hidden_dim * scale, hidden_dim, bias=True)

        self.drop_path = DropPath(drop_prob=drop_path)

    def forward(self, q, c):
        B, N, C, H, W = c.shape
        c = c.permute(0, 3, 4, 1, 2)
        c = self.norm(c)
        q = self.embed_q(q)
        q = self.proj_q(q)
        q = q.view(B, self.n_heads, self.head_dim)
        k, v = self.proj_kv(c).view(B, H, W, N, self.n_heads, self.head_dim * 2).permute(0, 1, 2, 4, 3, 5).chunk(2, -1)
        dots = q[:, None, None, :, None, :] @ k.transpose(-2, -1) * self.scale
        dots = F.softmax(dots, dim=-1)
        h = dots @ v
        h = self.proj_out(h.flatten(-3)).permute(0, 3, 1, 2)
        return h
