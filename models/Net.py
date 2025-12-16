import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from layers.layers import *
from utils import get_parameter_number

import os
from utils import parse_args


class Encoder(nn.Module):
    def __init__(self,
                 in_channel=1,
                 inner_channel=32,
                 norm_groups=32,
                 drop_out=0.1,
                 channel_mults=(1, 2, 4, 8),
                 with_attn=(8,),
                 drop_path=0.1,
                 ):
        super().__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel

        self.first_conv = nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        for ind in range(num_mults):
            channel_mult = inner_channel * channel_mults[ind]
            use_attn = True if channel_mults[ind] in with_attn else False
            self.downs.append(
                nn.Sequential(
                    BasicBlock(pre_channel, channel_mult, norm_groups=norm_groups, drop_out=drop_out,
                               with_attn=use_attn, drop_path=drop_path),
                    BasicBlock(channel_mult, channel_mult, norm_groups=norm_groups, drop_out=drop_out,
                               with_attn=use_attn, drop_path=drop_path),

                )
            )
            if ind != num_mults - 1:
                self.downs.append(DownSample(channel_mult))
            pre_channel = channel_mult

        self.mid = nn.Sequential(
            BasicBlock(pre_channel, pre_channel, norm_groups=norm_groups, drop_out=drop_out, with_attn=True,
                       drop_path=drop_path),
            BasicBlock(pre_channel, pre_channel, norm_groups=norm_groups, drop_out=drop_out, with_attn=True,
                       drop_path=drop_path),
        )

    def forward(self, x):
        conditions = []
        x = self.first_conv(x)
        for layer in self.downs:
            if isinstance(layer, nn.Sequential):
                x = layer[0](x)
                conditions.append(x)
                x = layer[1](x)
                conditions.append(x)
            else:
                x = layer(x)

        x = self.mid(x)
        conditions.append(x)
        return conditions


class UNet(nn.Module):
    def __init__(self,
                 in_channel=1,
                 inner_channel=32,
                 time_dim=32,
                 norm_groups=32,
                 drop_out=0.1,
                 channel_mults=(1, 2, 4, 8),
                 with_attn=(8,),
                 drop_path=0.1,
                 ):
        super().__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel

        self.first_conv = nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)
        self.downs = nn.ModuleList([])
        for ind in range(num_mults):
            channel_mult = inner_channel * channel_mults[ind]
            use_attn = True if channel_mults[ind] in with_attn else False
            self.downs.append(
                nn.Sequential(
                    ResnetBlock(pre_channel, channel_mult, time_dim=time_dim, norm_groups=norm_groups,
                                drop_out=drop_out,
                                condition_dim=channel_mult,
                                with_attn=use_attn,
                                drop_path=drop_path,
                                ),
                    ResnetBlock(channel_mult, channel_mult, time_dim=time_dim, norm_groups=norm_groups,
                                drop_out=drop_out,
                                condition_dim=channel_mult,
                                with_attn=use_attn,
                                drop_path=drop_path,
                                ),

                )
            )
            if ind != num_mults - 1:
                self.downs.append(DownSample(channel_mult))
            pre_channel = channel_mult

        self.mid = nn.ModuleList([
            ResnetBlock(pre_channel, pre_channel, time_dim=time_dim, norm_groups=norm_groups,
                        drop_out=drop_out,
                        condition_dim=pre_channel,
                        with_attn=True,
                        drop_path=drop_path,
                        ),
            ResnetBlock(pre_channel, pre_channel, time_dim=time_dim, norm_groups=norm_groups,
                        drop_out=drop_out,
                        condition_dim=0,
                        with_attn=True,
                        drop_path=drop_path,
                        ),
        ])

        self.ups = nn.ModuleList([])

        for ind in reversed(range(num_mults)):
            channel_mult = inner_channel * channel_mults[ind]
            use_attn = True if channel_mults[ind] in with_attn else False
            self.ups.append(
                nn.Sequential(
                    ResnetBlock(pre_channel + channel_mult, channel_mult,
                                time_dim=time_dim,
                                norm_groups=norm_groups,
                                drop_out=drop_out,
                                condition_dim=0,
                                with_attn=use_attn,
                                drop_path=drop_path,
                                ),
                    ResnetBlock(channel_mult + channel_mult, channel_mult,
                                time_dim=time_dim,
                                norm_groups=norm_groups,
                                drop_out=drop_out,
                                condition_dim=0,
                                with_attn=use_attn,
                                drop_path=drop_path,
                                ),
                )
            )
            if ind != 0:
                self.ups.append(UpSample(channel_mult, channel_mult))
            pre_channel = channel_mult
        self.final_conv = nn.Conv2d(inner_channel, in_channel, 3, 1, 1)

    def forward(self, x, t=None, conditions=None):
        feats = []
        x = self.first_conv(x)
        for layer in self.downs:
            if isinstance(layer, nn.Sequential):
                x = layer[0](x, t, conditions.pop())
                feats.append(x)
                x = layer[1](x, t, conditions.pop())
                feats.append(x)
            else:
                x = layer(x)
        x = self.mid[0](x, t, conditions.pop())
        x = self.mid[1](x, t)

        for layer in self.ups:
            if isinstance(layer, nn.Sequential):
                x = layer[0](op(x, feats.pop()), t)
                x = layer[1](op(x, feats.pop()), t)
            else:
                x = layer(x)
        x = self.final_conv(x)
        return x

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu'))
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu'))
            if m.bias is not None:
                init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            init.ones_(m.weight)
            init.zeros_(m.bias)


class Diffusion(nn.Module):
    def __init__(self,
                 in_channel=1,
                 inner_channel=32,
                 time_dim=int(32 * 4),
                 num_embeddings=5,
                 norm_groups=32,
                 drop_out=0.1,
                 channel_mults=(1, 2, 4, 8, 16),
                 with_attn=(8, 16),
                 drop_path=0.1,
                 ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            PositionalEncoding(time_dim),
        )
        self.encoder = Encoder(in_channel, inner_channel, norm_groups, drop_out, channel_mults, with_attn)
        self.fusion = nn.ModuleList([])

        for i in range(len(channel_mults)):
            self.fusion.append(
                Attention(num_embeddings, inner_channel * channel_mults[i], n_heads=channel_mults[i],
                          drop_path=drop_path))
            self.fusion.append(
                Attention(num_embeddings, inner_channel * channel_mults[i], n_heads=channel_mults[i],
                          drop_path=drop_path))
        self.fusion.append(
            Attention(num_embeddings, inner_channel * channel_mults[-1], n_heads=channel_mults[-1],
                      drop_path=drop_path))

        self.unet = UNet(in_channel, inner_channel, time_dim, norm_groups, drop_out, channel_mults, with_attn,
                         drop_path)
        self.conditions = None

    def forward(self, x, t, *conditions, embeddings=None, first_step=True):
        assert len(conditions) >= 2, 'num of image must more than 2'

        if embeddings is None:
            embeddings = torch.zeros(len(x), dtype=torch.long, device=x.device)
        elif len(embeddings) != len(x):
            embeddings = embeddings.repeat(len(x))

        # if self.training or first_step:
        if first_step:
            self.conditions = [self.encoder(c) for c in conditions]
            self.conditions = [torch.stack(c, dim=1) for c in zip(*self.conditions)]
            self.conditions = [f(embeddings, c) for f, c in zip(self.fusion, self.conditions)]
            self.conditions.reverse()

        t = self.time_mlp(t)
        out = self.unet(x, t, self.conditions.copy())
        return out


if __name__ == '__main__':
    t = torch.randint(1000, 1001, (4,))
    embeddings = torch.zeros(4).long()
    x = torch.randn(4, 1, 113, 113)

    args = parse_args()
    args.use_checkpoint = True if args.use_checkpoint == 'True' else False
    args.restart = True if args.restart == 'True' else False

    net = Diffusion()
    state_dict = torch.load('../logs/pretrain_udiff_ch1248A6_ep30k_best_loss.ckpt')
    print(state_dict['epoch'])
    print(state_dict['tf_logs_dir'])
    for k, v in state_dict.items():
        print(k)
    exit(0)
    net.load_state_dict(state_dict, strict=True)
    exit(0)
    for name, param in net.named_parameters():
        print(name)
    out = net(x, t, x, x, embeddings=embeddings)
    print(out.shape)
    get_parameter_number(net)
