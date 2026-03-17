import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from diffusion import create_diffusion


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        device = z.device
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


class SimpleMLPAdaLNFlow(nn.Module):
    """
    MLP for Flow Matching Loss (x-prediction, no learned variance).
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(model_channels))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        :param x: [N x C] noisy input.
        :param t: 1-D continuous timesteps in [0, 1].
        :param c: conditioning from AR transformer.
        :return: [N x C] x-prediction.
        """
        x = self.input_proj(x)
        # Scale t from [0,1] to [0,1000] for sinusoidal embedding
        t = self.time_embed(t * 1000.0)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale, t_eps=0.05):
        """CFG in v-space (velocity direction interpolation)."""
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        x_pred = self.forward(combined, t, c)

        # Convert x-prediction to v-prediction: v = (x_pred - z_t) / (1 - t)
        one_minus_t = (1.0 - t).clamp_min(t_eps).to(x_pred.dtype)
        if one_minus_t.dim() == 1:
            one_minus_t = one_minus_t.unsqueeze(-1)
        v_pred = (x_pred - combined) / one_minus_t

        v_cond, v_uncond = torch.split(v_pred, len(v_pred) // 2, dim=0)
        cfg_scale = torch.as_tensor(cfg_scale, dtype=x_pred.dtype, device=x_pred.device)
        guided_v = v_uncond + cfg_scale * (v_cond - v_uncond)
        guided_v = torch.cat([guided_v, guided_v], dim=0)

        return combined + one_minus_t * guided_v


class FlowDiffLoss(nn.Module):
    """Flow Matching Loss with x-prediction and v-loss (JiT-style)."""

    def __init__(self, target_channels, z_channels, depth, width,
                 num_sampling_steps=50, grad_checkpointing=False,
                 P_mean=-0.8, P_std=0.8, noise_scale=1.0,
                 t_eps=0.05, sampling_method='euler'):
        super(FlowDiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLNFlow(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels,  # no learned variance
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.P_mean = P_mean
        self.P_std = P_std
        self.noise_scale = noise_scale
        self.t_eps = t_eps
        self.num_sampling_steps = int(num_sampling_steps)
        self.sampling_method = sampling_method

    def forward(self, target, z, mask=None):
        bsz = target.shape[0]
        device = target.device

        # Logit-normal time sampling
        u = torch.randn(bsz, device=device) * self.P_std + self.P_mean
        t = torch.sigmoid(u)  # t in (0, 1)

        # Interpolate: z_t = t * target + (1 - t) * noise
        noise = torch.randn_like(target) * self.noise_scale
        t_expand = t.unsqueeze(-1)
        z_t = t_expand * target + (1.0 - t_expand) * noise

        # x-prediction
        x_pred = self.net(z_t, t, z)

        # v-loss: loss on velocity space
        # v_pred = (x_pred - z_t) / (1 - t), v_target = (target - z_t) / (1 - t)
        # MSE in v-space = MSE(x_pred, target) / (1 - t)^2
        one_minus_t = (1.0 - t_expand).clamp_min(self.t_eps)
        loss = ((x_pred - target) / one_minus_t) ** 2
        loss = loss.mean(dim=-1)  # mean over channels

        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        device = z.device
        N = self.num_sampling_steps
        dt = 1.0 / N

        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=device) * temperature
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg, t_eps=self.t_eps)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=device) * temperature
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        x_t = noise  # start at t=0 (pure noise)

        for i in range(N):
            t_cur = torch.full((x_t.shape[0],), i * dt, device=device, dtype=torch.float32)

            x_pred = sample_fn(x_t, t_cur, **model_kwargs)

            if self.sampling_method == 'euler':
                # velocity: v = (x_pred - x_t) / (1 - t)
                one_minus_t = (1.0 - t_cur).clamp_min(self.t_eps)
                if one_minus_t.dim() == 1:
                    one_minus_t = one_minus_t.unsqueeze(-1)
                v = (x_pred - x_t) / one_minus_t
                x_t = x_t + v * dt

            elif self.sampling_method == 'heun':
                one_minus_t = (1.0 - t_cur).clamp_min(self.t_eps)
                if one_minus_t.dim() == 1:
                    one_minus_t = one_minus_t.unsqueeze(-1)
                v1 = (x_pred - x_t) / one_minus_t
                x_t_next = x_t + v1 * dt

                if i < N - 1:
                    t_next = torch.full((x_t.shape[0],), (i + 1) * dt, device=device, dtype=torch.float32)
                    x_pred_next = sample_fn(x_t_next, t_next, **model_kwargs)
                    one_minus_t_next = (1.0 - t_next).clamp_min(self.t_eps)
                    if one_minus_t_next.dim() == 1:
                        one_minus_t_next = one_minus_t_next.unsqueeze(-1)
                    v2 = (x_pred_next - x_t_next) / one_minus_t_next
                    x_t = x_t + 0.5 * (v1 + v2) * dt
                else:
                    x_t = x_t_next
            else:
                raise ValueError(f"Unknown sampling method: {self.sampling_method}")

        return x_t
