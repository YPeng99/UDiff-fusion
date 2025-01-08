import math
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms

from utils import denormalizer


class Gaussian:
    def __init__(self, time_steps=1000, beta_schedule='linear', linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3,
                 device=torch.device('cuda')):
        self.time_steps = time_steps
        self.beta_schedule = beta_schedule
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.cosine_s = cosine_s
        self.device = device
        self.betas = self.make_beta_schedule().to(device)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        self.device = device
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)

    def _warmup_beta(self, linear_start, linear_end, n_timestep, warmup_frac):
        betas = linear_end * torch.ones(n_timestep)
        warmup_time = int(n_timestep * warmup_frac)
        betas[:warmup_time] = torch.linspace(
            linear_start, linear_end, warmup_time)
        return betas

    def make_beta_schedule(self):
        if self.beta_schedule == 'quad':
            betas = torch.linspace(self.linear_start ** 0.5, self.linear_end ** 0.5, self.time_steps) ** 2
        elif self.beta_schedule == 'linear':
            betas = torch.linspace(self.linear_start, self.linear_end, self.time_steps)
        elif self.beta_schedule == 'warmup10':
            betas = self._warmup_beta(self.linear_start, self.linear_end, self.time_steps, 0.1)
        elif self.beta_schedule == 'warmup50':
            betas = self._warmup_beta(self.linear_start, self.linear_end, self.time_steps, 0.5)
        elif self.beta_schedule == 'const':
            betas = self.linear_end * torch.ones(self.time_steps)
        elif self.beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1. / torch.linspace(self.time_steps, 1, self.time_steps)
        elif self.beta_schedule == "cosine":
            timesteps = (
                    torch.arange(self.time_steps + 1, dtype=torch.float64) /
                    self.time_steps + self.cosine_s
            )
            alphas = timesteps / (1 + self.cosine_s) * math.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = betas.clamp(max=0.999)
        else:
            raise NotImplementedError(self.beta_schedule)
        return betas

    def get_index_from_list(self, vals, time_step, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = time_step.shape[0]
        out = vals.gather(-1, time_step)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def forward_diffusion_sample(self, x_0, time_step):
        """
        Takes an image and a timestep as input and
        returns the noisy version of it
        """
        time_step = time_step.to(self.device)
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, time_step, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, time_step,
                                                                   x_0.shape)

        # mean + variance
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.inference_mode()
    def ddpm_sample(self, diff,*conditions,embeddings,clip_denoised=True):
        diff.eval()
        sample_images = torch.randn_like(conditions[0])
        first_step = True
        for t in reversed(range(0, self.time_steps)):
            t = torch.full((sample_images.shape[0],), t, device=self.device, dtype=torch.long)
            betas_t = self.get_index_from_list(self.betas, t, sample_images.shape)
            sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t,
                                                                       sample_images.shape)
            sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, sample_images.shape)

            sample_images = sqrt_recip_alphas_t *(sample_images - betas_t * diff(sample_images, t,*conditions,embeddings=embeddings,first_step=first_step) / sqrt_one_minus_alphas_cumprod_t)

            if clip_denoised:
                sample_images = torch.clamp(sample_images, -1, 1)
            first_step = False

            # if t[0] != 0:
            #     noise = torch.randn_like(sample_images)
            #     posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, sample_images.shape)
            #     sample_images = sample_images + torch.sqrt(posterior_variance_t) * noise

        return sample_images

    @torch.inference_mode()
    def ddim_sample(self, diff,*conditions,embeddings, ddim_timesteps=30,
                    ddim_discr_method='uniform', ddim_eta=0.0, clip_denoised=True):
        diff.eval()
        if ddim_discr_method == 'uniform':
            c = self.time_steps // ddim_timesteps
            ddim_timestep_seq = torch.arange(0, self.time_steps, c)
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (torch.linspace(0, torch.sqrt(torch.tensor(self.time_steps)) * .8, ddim_timesteps)) ** 2

        else:
            raise NotImplementedError(f"There is no ddim discretization method called {ddim_discr_method}")

        ddim_timestep_seq = ddim_timestep_seq.type(torch.int64) + 1
        ddim_timestep_prev_seq = torch.cat([torch.tensor([0]), ddim_timestep_seq[:-1]])

        sample_images = torch.randn_like(conditions[0])
        first_step = True
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((sample_images.shape[0],), int(ddim_timestep_seq[i]), device=self.device, dtype=torch.long)
            prev_t = torch.full((sample_images.shape[0],), int(ddim_timestep_prev_seq[i]), device=self.device, dtype=torch.long)

            alpha_cumprod_t = self.get_index_from_list(self.alphas_cumprod, t, sample_images.shape)
            alpha_cumprod_t_prev = self.get_index_from_list(self.alphas_cumprod, prev_t, sample_images.shape)

            pred_noise = diff(sample_images, t,*conditions,embeddings=embeddings,first_step=first_step)
            pred_x0 = (sample_images - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)

            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, -1, 1)

            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(
                sample_images)

            sample_images = x_prev
            first_step = False
        return sample_images



