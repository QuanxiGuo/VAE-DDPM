# diffusion_model_architecture.py
import torch
import torch.nn as nn
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, cond_emb_dim):
        super().__init__()
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        if out_channels >= 8:
            self.norm1 = nn.GroupNorm(min(8, out_channels), out_channels)
            self.norm2 = nn.GroupNorm(min(8, out_channels), out_channels)
        else:
            self.norm1 = nn.LayerNorm(out_channels)
            self.norm2 = nn.LayerNorm(out_channels)

        self.activation = nn.SiLU()

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, cond_emb):
        h = self.conv1(x)

        if isinstance(self.norm1, nn.GroupNorm):
            h = self.norm1(h)
        else:
            h = h.transpose(1, 2)
            h = self.norm1(h)
            h = h.transpose(1, 2)

        h = self.activation(h)

        cond_emb_projected = self.activation(self.cond_mlp(cond_emb))
        h = h + cond_emb_projected[:, :, None]

        h = self.conv2(h)

        if isinstance(self.norm2, nn.GroupNorm):
            h = self.norm2(h)
        else:
            h = h.transpose(1, 2)
            h = self.norm2(h)
            h = h.transpose(1, 2)

        h = self.activation(h)

        return h + self.residual_conv(x)


class UNet1DConditional(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128,
                 dqdv_latent_dim=16, max_cycle_num=51, cycle_emb_dim=16,
                 features=[64, 128, 256]):
        super().__init__()

        self.time_emb_dim = time_emb_dim
        self.dqdv_latent_dim = dqdv_latent_dim
        self.cycle_emb_dim = cycle_emb_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.cycle_embedding = nn.Embedding(max_cycle_num + 1, cycle_emb_dim)

        self.dqdv_latent_project = nn.Linear(dqdv_latent_dim, cycle_emb_dim)

        self.cond_emb_dim = time_emb_dim + cycle_emb_dim
        self.combine_cond_mlp = nn.Sequential(
            nn.Linear(time_emb_dim + cycle_emb_dim, self.cond_emb_dim),
            nn.ReLU(),
            nn.Linear(self.cond_emb_dim, self.cond_emb_dim)
        )

        self.init_conv = nn.Conv1d(in_channels, features[0], 3, padding=1)

        self.encoder = nn.ModuleList()
        self.encoder.append(ResidualBlock(features[0], features[0], self.cond_emb_dim))
        for i in range(1, len(features)):
            self.encoder.append(ResidualBlock(features[i - 1], features[i], self.cond_emb_dim))

        self.bottleneck = ResidualBlock(features[-1], features[-1], self.cond_emb_dim)

        self.decoder = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoder.append(ResidualBlock(features[i] + features[i - 1], features[i - 1], self.cond_emb_dim))

        self.final_conv = nn.Conv1d(features[0], out_channels, 1)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x, timestep, dqdv_latent, cycle_num):
        time_emb = self.time_mlp(timestep)
        cycle_emb = self.cycle_embedding(cycle_num).squeeze(1)
        dqdv_proj = self.dqdv_latent_project(dqdv_latent)

        combined_cycle_dqdv_emb = cycle_emb + dqdv_proj

        cond_emb = self.combine_cond_mlp(torch.cat([time_emb, combined_cycle_dqdv_emb], dim=-1))

        h = self.init_conv(x)

        encoder_features = []
        for i, encoder_layer in enumerate(self.encoder):
            h = encoder_layer(h, cond_emb)
            encoder_features.append(h)
            if i < len(self.encoder) - 1:
                h = self.pool(h)

        h = self.bottleneck(h, cond_emb)

        for i, decoder_layer in enumerate(self.decoder):
            h = nn.functional.interpolate(h, scale_factor=2, mode='linear', align_corners=False)
            skip_connection = encoder_features[len(self.encoder) - 2 - i]

            if h.size(-1) != skip_connection.size(-1):
                h = nn.functional.interpolate(h, size=skip_connection.size(-1),
                                              mode='linear', align_corners=False)

            h = torch.cat([h, skip_connection], dim=1)
            h = decoder_layer(h, cond_emb)

        return self.final_conv(h)