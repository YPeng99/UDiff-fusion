import torch
import torch.nn as nn

from layers.RMSNorm import RMSNorm
import torch.nn.functional as F
import math


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim, dim_out, 3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class BasicBlock(nn.Module):
    def __init__(self, dim, dim_out, norm_groups=32, dropout=0.1, with_attn=False):
        super().__init__()
        self.block_1 = nn.Sequential(
            Block(dim, dim_out, groups=norm_groups, dropout=dropout),
        )

        self.block_2 = nn.Sequential(
            Block(dim_out, dim_out, groups=norm_groups, dropout=dropout),
            # Block(dim_out, dim_out, groups=norm_groups, dropout=dropout),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.attn = SelfAttention(dim_out, norm_groups=norm_groups) if with_attn else None

    def forward(self, x):
        h = self.block_1(x)
        h = self.block_2(h) + self.res_conv(x)
        if self.attn is not None:
            h = self.attn(h) + h
        return h


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, norm_groups=32, dropout=0.1, time_dim=32, condition_dim=0, with_attn=False):
        super().__init__()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.time_mlp = nn.Linear(time_dim, dim_out * 2)
        dim = dim + condition_dim
        self.block_1 = nn.Sequential(
            Block(dim, dim_out, groups=norm_groups, dropout=dropout),
        )
        self.block_2 = nn.Sequential(
            Block(dim_out, dim_out, groups=norm_groups, dropout=dropout),
            # Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        )

        self.attn = SelfAttention(dim_out, norm_groups=norm_groups) if with_attn else None

    def forward(self, x, t, condition=None):
        h = x
        if condition is not None:
            h = torch.cat([x, condition], dim=1)
        h = self.block_1(h)
        gamma, beta = self.time_mlp(t).view(h.shape[0], -1, 1, 1).chunk(2, 1)
        # t = t.view(h.shape[0], -1, 1, 1)
        # h = h + t

        h = (1 + gamma) * h + beta
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
    def __init__(self, num_embeddings=2, hidden_dim=32, scale=1, n_heads=1):
        super().__init__()
        self.n_heads = n_heads
        self.head_size = hidden_dim * scale // self.n_heads
        self.scale = self.head_size ** -0.5
        self.norm = RMSNorm(hidden_dim)

        self.proj_q = nn.Sequential(
            nn.Embedding(num_embeddings, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * scale),
        )
        self.proj_kv = nn.Linear(hidden_dim, hidden_dim * scale * 2, bias=False)
        self.proj_out = nn.Linear(hidden_dim * scale, hidden_dim, bias=True)

    def forward(self, q, c):
        B, N, C, H, W = c.shape
        c = c.permute(0, 3, 4, 1, 2)
        c = self.norm(c)
        q = self.proj_q(q)
        k, v = self.proj_kv(c).chunk(2, dim=-1)
        q = q.view(B, self.n_heads, self.head_size)
        k = k.view(B, H, W, N, self.n_heads, self.head_size).permute(0, 1, 2, 4, 5, 3)
        v = v.view(B, H, W, N, self.n_heads, self.head_size).permute(0, 1, 2, 4, 3, 5)
        dots = torch.matmul(q[:, None, None, :, None, :], k) * self.scale
        dots = F.softmax(dots, dim=-1)
        h = self.proj_out(torch.matmul(dots, v).flatten(-3)).permute(0, 3, 1, 2)
        return h
