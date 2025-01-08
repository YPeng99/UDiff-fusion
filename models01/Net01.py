import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sympy.strategies.branch import condition

from layers.layers import PositionalEncoding, DownSample, UpSample, op
from models01.layer01 import ResnetBlock, BasicBlock, Attention,Block
from utils import get_parameter_number


class Encoder(nn.Module):
    def __init__(self,
                 in_channel=1,
                 inner_channel=32,
                 norm_groups=32,
                 dropout=0.1,
                 channel_mults=(1, 2, 4, 8, 8),
                 with_atten=(8,),
                 ):
        super().__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel

        self.first_conv = nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        self.feats_channel = []
        for ind in range(num_mults):
            channel_mult = inner_channel * channel_mults[ind]
            use_attn = True if channel_mults[ind] in with_atten else False
            # self.feats_channel.append(pre_channel)
            # self.feats_channel.append(channel_mult)
            self.downs.append(
                nn.Sequential(
                    BasicBlock(pre_channel, pre_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn),
                    BasicBlock(pre_channel, channel_mult, norm_groups=norm_groups, dropout=dropout,
                               with_attn=use_attn),
                    # BasicBlock(channel_mult, channel_mult, norm_groups=norm_groups, dropout=dropout,
                    #            with_attn=use_attn),
                )
            )
            if ind != num_mults - 1:
                self.downs.append(DownSample(channel_mult))
            pre_channel = channel_mult

        self.mid = BasicBlock(pre_channel, pre_channel, norm_groups=norm_groups, dropout=dropout, with_attn=True)

        # self.ups = nn.ModuleList([])
        #
        # for ind in reversed(range(num_mults)):
        #     channel_mult = inner_channel * channel_mults[ind]
        #     use_attn = True if channel_mults[ind] in with_atten else False
        #     self.ups.append(
        #         nn.Sequential(
        #             BasicBlock(pre_channel + self.feats_channel.pop(), channel_mult, norm_groups=norm_groups,
        #                        dropout=dropout,
        #                        with_attn=use_attn),
        #             BasicBlock(channel_mult + self.feats_channel.pop(), channel_mult, norm_groups=norm_groups,
        #                        dropout=dropout,
        #                        with_attn=use_attn),
        #         )
        #     )
        #
        #     if ind != 0:
        #         self.ups.append(UpSample(channel_mult, channel_mult))
        #     pre_channel = channel_mult

    def forward(self, x):
        feats = []
        x = self.first_conv(x)
        for layer in self.downs:
            if isinstance(layer, nn.Sequential):
                x = layer[0](x)
                feats.append(x)
                x = layer[1](x)
                feats.append(x)
                # x = layer[2](x)
                # feats.append(x)
            else:
                x = layer(x)

        conditions = feats.copy()
        x = self.mid(x)
        conditions.append(x)
        # x = self.mid[1](x)
        # conditions.append(x)
        #
        # for layer in self.ups:
        #     if isinstance(layer, nn.Sequential):
        #         x = layer[0](op(x, feats.pop()))
        #         conditions.append(x)
        #         x = layer[1](op(x, feats.pop()))
        #         conditions.append(x)
        #     else:
        #         x = layer(x)
        return conditions


class UNet(nn.Module):
    def __init__(self,
                 in_channel=1,
                 inner_channel=32,
                 time_dim=32,
                 norm_groups=32,
                 dropout=0.1,
                 channel_mults=(1, 2, 4, 6, 8),
                 with_atten=(8,),
                 ):
        super().__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel

        self.first_conv = nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        self.feats_channel = []
        for ind in range(num_mults):
            channel_mult = inner_channel * channel_mults[ind]
            use_attn = True if channel_mults[ind] in with_atten else False
            self.feats_channel.append(channel_mult)
            self.feats_channel.append(channel_mult)
            # self.feats_channel.append(channel_mult)
            self.downs.append(
                nn.Sequential(
                    ResnetBlock(pre_channel, channel_mult, time_dim=time_dim, norm_groups=norm_groups,
                                dropout=dropout,
                                condition_dim=pre_channel,
                                with_attn=use_attn),
                    ResnetBlock(channel_mult, channel_mult, time_dim=time_dim, norm_groups=norm_groups,
                                dropout=dropout,
                                condition_dim=channel_mult,
                                with_attn=use_attn),
                    # ResnetBlock(channel_mult, channel_mult, time_dim=time_dim, norm_groups=norm_groups,
                    #             dropout=dropout,
                    #             condition_dim=channel_mult,
                    #             with_attn=use_attn),
                )
            )
            if ind != num_mults - 1:
                self.downs.append(DownSample(channel_mult))
            pre_channel = channel_mult

        self.mid = nn.ModuleList([
            ResnetBlock(pre_channel, pre_channel, time_dim=time_dim, norm_groups=norm_groups,
                        dropout=dropout,
                        condition_dim=pre_channel,
                        with_attn=True),
            ResnetBlock(pre_channel, pre_channel, time_dim=time_dim, norm_groups=norm_groups,
                        dropout=dropout,
                        condition_dim=0,
                        with_attn=False),
        ])

        self.ups = nn.ModuleList([])

        for ind in reversed(range(num_mults)):
            channel_mult = inner_channel * channel_mults[ind]
            use_attn = True if channel_mults[ind] in with_atten else False
            self.ups.append(
                nn.Sequential(
                    ResnetBlock(pre_channel + self.feats_channel.pop(), channel_mult, time_dim=time_dim,
                                norm_groups=norm_groups,
                                dropout=dropout,
                                condition_dim=0,
                                with_attn=use_attn),
                    ResnetBlock(channel_mult + self.feats_channel.pop(), channel_mult, time_dim=time_dim,
                                norm_groups=norm_groups,
                                dropout=dropout,
                                condition_dim=0,
                                with_attn=use_attn),
                    # ResnetBlock(channel_mult + self.feats_channel.pop(), channel_mult, time_dim=time_dim,
                    #             norm_groups=norm_groups,
                    #             dropout=dropout,
                    #             condition_dim=0,
                    #             with_attn=use_attn)
                )
            )
            if ind != 0:
                self.ups.append(UpSample(channel_mult, channel_mult))
            pre_channel = channel_mult

    def forward(self, x, t, conditions):
        feats = []
        x = self.first_conv(x)
        for layer in self.downs:
            if isinstance(layer, nn.Sequential):
                x = layer[0](x, t, conditions.pop())
                feats.append(x)
                x = layer[1](x, t, conditions.pop())
                feats.append(x)
                # x = layer[2](x, t, conditions.pop())
                # feats.append(x)
            else:
                x = layer(x)

        x = self.mid[0](x, t, conditions.pop())
        x = self.mid[1](x, t)

        for layer in self.ups:
            if isinstance(layer, nn.Sequential):
                x = layer[0](op(x, feats.pop()), t)
                x = layer[1](op(x, feats.pop()), t)
                # x = layer[2](op(x, feats.pop()), t)
            else:
                x = layer(x)

        return x


class Diffusion(nn.Module):
    def __init__(self,
                 in_channel=1,
                 inner_channel=32,
                 time_dim=64,
                 num_embeddings=5,
                 norm_groups=32,
                 dropout=0.1,
                 channel_mults=(1, 2, 4, 8),
                 with_atten=(8,),
                 ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            PositionalEncoding(time_dim),
            nn.Linear(time_dim, time_dim*4),
            nn.SiLU(),
            nn.Linear(time_dim*4, time_dim),
        )
        self.encoder = Encoder(in_channel, inner_channel, norm_groups, dropout, channel_mults, with_atten)
        self.unet = UNet(in_channel, inner_channel, time_dim, norm_groups, dropout, channel_mults, with_atten)
        self.attn = nn.ModuleList([])

        self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[0], n_heads=channel_mults[0]))
        self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[0], n_heads=channel_mults[0]))
        # self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[0], n_heads=channel_mults[0]))
        for i in range(len(channel_mults) - 1):
            self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[i], n_heads=channel_mults[i]))
            self.attn.append(
                Attention(num_embeddings, inner_channel * channel_mults[i + 1], n_heads=channel_mults[i + 1]))
            # self.attn.append(
            #     Attention(num_embeddings, inner_channel * channel_mults[i + 1], n_heads=channel_mults[i + 1]))
        self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[-1], n_heads=channel_mults[-1]))
        # self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[-1], n_heads=channel_mults[-1]))

        # for i in range(len(channel_mults)):
        #     self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[i], n_heads=channel_mults[i]))
        #     self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[i], n_heads=channel_mults[i]))
        # self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[-1], n_heads=channel_mults[-1]))
        # self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[-1], n_heads=channel_mults[-1]))
        # for i in reversed(range(len(channel_mults))):
        #     self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[i], n_heads=channel_mults[i]))
        #     self.attn.append(Attention(num_embeddings, inner_channel * channel_mults[i], n_heads=channel_mults[i]))
        self.final_block = nn.Sequential(
            Block(inner_channel,in_channel),
        )
        self.conditions = None

    def forward(self, x, t, *conditions, embeddings=None, first_step=True):
        assert len(conditions) >= 2, 'num of image must more than 2'

        if embeddings is None:
            embeddings = torch.zeros(len(x), dtype=torch.long, device=x.device)
        elif len(embeddings) != len(x):
            embeddings = embeddings.repeat(len(x))

        if self.training:
            self.conditions = [self.encoder(c) for c in conditions]
            self.conditions = [torch.stack(c, dim=1) for c in zip(*self.conditions)]
            self.conditions = [attn(embeddings, c) for attn, c in zip(self.attn, self.conditions)]
            self.conditions.reverse()

        else:
            if first_step:
                self.conditions = [self.encoder(c) for c in conditions]
                self.conditions = [torch.stack(c, dim=1) for c in zip(*self.conditions)]
                self.conditions = [attn(embeddings, c) for attn, c in zip(self.attn, self.conditions)]
                self.conditions.reverse()

        t = self.time_mlp(t)
        out = self.unet(x, t, self.conditions.copy())
        out = self.final_block(out)
        return out


if __name__ == '__main__':
    t = torch.randint(1000, 1001, (4,))
    embeddings = torch.zeros(4).long()
    x = torch.randn(4, 1, 112, 112)
    diff = Diffusion()
    out = diff(x, t, x, x, embeddings=embeddings)
    get_parameter_number(diff)
