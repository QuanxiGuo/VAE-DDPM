
import torch
import torch.nn as nn
import numpy as np

class DiffusionModelConditional:

    def __init__(self, model, num_timesteps=300, beta_start=1e-3, beta_end=0.02, device=None):
        self.model = model
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, dqdv_latent, cycle_num, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t, dqdv_latent, cycle_num)

        loss = nn.functional.mse_loss(noise, predicted_noise)
        return loss

    def p_sample(self, shape, target_dqdv_latents: torch.Tensor, target_cycle_numbers: torch.Tensor):
        b = shape[0]
        img = torch.randn(shape, device=self.device)

        target_dqdv_latents = target_dqdv_latents.to(self.device)
        target_cycle_numbers = target_cycle_numbers.to(self.device)

        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self.p_sample_step(img, t, target_dqdv_latents, target_cycle_numbers)

        return img

    def p_sample_step(self, x, t, dqdv_latent, cycle_num):
        betas_t = self.betas[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None, None]

        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t, dqdv_latent, cycle_num) / sqrt_one_minus_alphas_cumprod_t
        )

        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise