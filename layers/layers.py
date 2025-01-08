import torch
import torch.nn as nn
from layers.RMSNorm import RMSNorm
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
        # self.down = nn.AvgPool2d(3,2,1)

    def forward(self, x):
        return self.down(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    def __init__(self, dim, dim_out, norm_groups=32):
        super().__init__()
        self.block_1 = Block(dim, dim_out, groups=norm_groups)

        self.block_2 = nn.Sequential(
            Block(dim_out, dim_out, groups=norm_groups),
            Block(dim_out, dim_out, groups=norm_groups)
        )
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block_1(x)
        h = self.block_2(h) + self.res_conv(x)
        return h


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, norm_groups=32, time_dim=None, add_condition=False, with_attn=False):
        super().__init__()
        self.time_mlp = nn.Identity() if time_dim is None else nn.Linear(time_dim, dim)
        self.control = Attention(dim, scale=2) if add_condition else nn.Identity()
        self.query = nn.Embedding(2, dim) if add_condition else nn.Identity()

        self.block_1 = Block(dim * 2, dim_out, groups=norm_groups) if add_condition else Block(dim, dim_out,
                                                                                               groups=norm_groups)
        self.block_2 = nn.Sequential(
            Block(dim_out, dim_out, groups=norm_groups),
            Block(dim_out, dim_out, groups=norm_groups)
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.attn = SelfAttention(dim_out, norm_groups=norm_groups) if with_attn else None

    def forward(self, x, t=None, conditions=None, embeddings=None):
        h = x
        if t is not None:
            t = self.time_mlp(t)
            h = h + t.view(x.shape[0], -1, 1, 1)

        if conditions is not None:
            q = self.query(embeddings) + t
            c = self.control(q, conditions)
            h = torch.cat([h, c], dim=1)
        h = self.block_1(h)
        h = self.block_2(h) + self.res_conv(x)
        if self.attn is not None:
            h = self.attn(h) + h
        return h


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out


class Attention(nn.Module):
    def __init__(self, embed_dim=32, scale=2, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = embed_dim * scale // self.n_heads
        self.scale = self.head_size ** -0.5
        self.norm_1 = RMSNorm(embed_dim)

        self.proj_q = nn.Linear(embed_dim, embed_dim * scale, bias=False)
        self.proj_kv = nn.Linear(embed_dim, embed_dim * scale * 2, bias=False)
        self.proj_out = nn.Linear(embed_dim * scale, embed_dim, bias=True)

        # self.norm_2 = nn.GroupNorm(num_groups=32, num_channels=embed_dim * 2)
        # self.ffn = nn.Sequential(
        #     nn.Conv2d(embed_dim * 2, embed_dim * 2, 3, 1, 1),
        #     nn.SiLU(),
        #     nn.Conv2d(embed_dim * 2, embed_dim, 3, 1, 1)
        # )

    def forward(self, q, c):
        B, N, C, H, W = c.shape
        c = c.permute(0, 3, 4, 1, 2)
        c = self.norm_1(c)
        q = self.proj_q(q)
        k, v = self.proj_kv(c).chunk(2, dim=-1)
        q = q.view(B, self.n_heads, self.head_size)
        k = k.view(B, H, W, N, self.n_heads, self.head_size).permute(0, 1, 2, 4, 5, 3)
        v = v.view(B, H, W, N, self.n_heads, self.head_size).permute(0, 1, 2, 4, 3, 5)
        dots = torch.matmul(q[:, None, None, :, None, :], k) * self.scale
        dots = F.softmax(dots, dim=-1)
        h = self.proj_out(torch.matmul(dots, v)).flatten(-3).permute(0, 3, 1, 2)

        # h = torch.cat([h,c],dim=1)
        # h = self.norm_2(h)
        # h = self.ffn(h)
        return h

#
