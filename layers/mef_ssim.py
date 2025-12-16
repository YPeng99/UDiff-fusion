from math import exp

import torch
import torch.nn.functional as F



def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / (gauss.sum())


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, window_size / 6.).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(1, channel, window_size, window_size) / channel
    return window


def _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=False, cen_lum=0., full=False):
    K, C, H, W = list(Ys.size())

    # compute statistics of the reference latent image Y
    muY_seq = F.conv2d(Ys, window, padding=ws // 2).view(K, H, W)
    muY_sq_seq = muY_seq * muY_seq
    sigmaY_sq_seq = F.conv2d(Ys * Ys, window, padding=ws // 2).view(K, H, W) \
                    - muY_sq_seq
    sigmaY_sq, patch_index = torch.max(sigmaY_sq_seq, dim=0)

    # compute statistics of the test image X
    muX = F.conv2d(X, window, padding=ws // 2).view(H, W)
    muX_sq = muX * muX
    sigmaX_sq = F.conv2d(X * X, window, padding=ws // 2).view(H, W) - muX_sq

    # compute correlation term
    sigmaXY = F.conv2d(X.expand_as(Ys) * Ys, window, padding=ws // 2).view(K, H, W) \
              - muX.expand_as(muY_seq) * muY_seq

    # compute quality map
    cs_seq = (2 * sigmaXY + C2) / (sigmaX_sq + sigmaY_sq_seq + C2)
    cs_map = torch.gather(cs_seq.view(K, -1), 0, patch_index.view(1, -1)).view(H, W)
    if is_lum:
        lY = torch.mean(muY_seq.view(K, -1), dim=1)
        lL = torch.exp(-((muY_seq - cen_lum) ** 2) / denom_l)
        lG = torch.exp(- ((lY - cen_lum) ** 2) / denom_g)[:, None, None].expand_as(lL)
        LY = lG * lL
        muY = torch.sum((LY * muY_seq), dim=0) / torch.sum(LY, dim=0)
        muY_sq = muY * muY
        l_map = (2 * muX * muY + C1) / (muX_sq + muY_sq + C1)
    else:
        l_map = torch.ones_like(cs_map)
    if full:
        l = torch.mean(l_map)
        cs = torch.mean(cs_map)
        return l, cs

    qmap = l_map * cs_map
    q = qmap.mean()

    return q


def mef_ssim(X, Ys, window_size=11, max_val=1.0, is_lum=False, cen_lum=0.):
    (_, channel, _, _) = Ys.size()
    window = create_window(window_size, channel).to(Ys)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    return _mef_ssim(X, Ys, window, window_size, 0.08, 0.08, C1, C2, is_lum, cen_lum)


def _mef_msssim(X, Ys, window, beta, ws, denom_g, denom_l, C1, C2, is_lum=False, cen_lum=0.):
    levels = beta.size()[0]
    l_i = []
    cs_i = []
    for _ in range(levels):
        l, cs = _mef_ssim(X, Ys, window, ws, denom_g, denom_l, C1, C2, is_lum=is_lum, cen_lum=cen_lum, full=True)
        l_i.append(l)
        cs_i.append(F.relu(cs))

        X = F.avg_pool2d(X, (2, 2))
        Ys = F.avg_pool2d(Ys, (2, 2))

    Ql = torch.stack(l_i)
    Qcs = torch.stack(cs_i)
    return (Ql[levels - 1] ** beta[levels - 1]) * torch.prod(Qcs ** beta)


def mef_mssim(X, Ys, window_size=11, beta=None, max_val=1.0, is_lum=False, cen_lum=0.):
    (_, channel, _, _) = Ys.size()
    window = create_window(window_size, channel).to(Ys)
    if beta is None:
        beta = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(Ys)
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    return _mef_msssim(X, Ys, window, beta, window_size, 0.08, 0.08, C1, C2, is_lum, cen_lum)


class MEF_SSIM_Loss(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, max_val=2.0, dtype=torch.float32):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        window = create_window(window_size, self.channel).type(dtype)
        self.register_buffer('window', window)
        self.denom_g = 2 * sigma_g ** 2
        self.denom_l = 2 * sigma_l ** 2
        self.C1 = (0.01 * max_val) ** 2
        self.C2 = (0.03 * max_val) ** 2

    def forward(self, X, Ys, is_lum=False, cen_lum=0.):
        return 1 - _mef_ssim(X, Ys, self.window, self.window_size,
                             self.denom_g, self.denom_l, self.C1, self.C2, is_lum, cen_lum)


class MEF_MSSSIM_Loss(torch.nn.Module):
    def __init__(self, window_size=11, channel=3, sigma_g=0.2, sigma_l=0.2, max_val=2.0, is_lum=False, cen_lum=0.,
                 dtype=torch.float32):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        window = create_window(window_size, self.channel).type(dtype)
        self.register_buffer('window', window)
        # beta = torch.tensor([0.0710, 0.4530, 0.4760])
        beta = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).type(dtype)
        # beta = torch.tensor([1, 1, 1, 1, 1])
        # beta = torch.tensor([1])
        self.register_buffer('beta', beta)
        self.denom_g = 2 * sigma_g ** 2
        self.denom_l = 2 * sigma_l ** 2
        self.C1 = (0.01 * max_val) ** 2
        self.C2 = (0.03 * max_val) ** 2
        self.is_lum = is_lum
        self.cen_lum = cen_lum

    def forward(self, X, Ys):
        return 1 - _mef_msssim(X, Ys, self.window, self.beta, self.window_size,
                               self.denom_g, self.denom_l, self.C1, self.C2, self.is_lum, self.cen_lum)
