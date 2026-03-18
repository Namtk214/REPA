import torch
import numpy as np


def compute_block_cosine_matrix(block_tokens):
    """
    Compute pairwise cosine similarity matrix between outputs of all DiT blocks.

    Args:
        block_tokens: list of L tensors, each of shape [B, N, D]
                      (output token of each block for the same batch)
    Returns:
        sim_mat: [L, L] CPU tensor, sim_mat[a, b] = mean cosine similarity
                 between block a and block b averaged over all patches and batch items.
    """
    H = torch.stack(block_tokens, dim=0).float()  # [L, B, N, D]

    # Normalize along hidden dimension (with epsilon for numerical stability)
    H_norm = H / (torch.linalg.norm(H, dim=-1, keepdim=True) + 1e-8)

    # Pairwise cosine via einsum: (L, B, N, D) x (L, B, N, D) -> (L, L, B, N)
    cosine_vals = torch.einsum('ibnd,jbnd->ijbn', H_norm, H_norm)

    # Average over batch and tokens: (L, L, B, N) -> (L, L)
    sim_mat = cosine_vals.mean(dim=(-2, -1))

    return sim_mat.cpu()


def expand_t_like_x(t, x_cur):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x_cur.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t

def get_score_from_velocity(vt, xt, t, path_type="linear"):
    """Wrapper function: transfrom velocity prediction model to score
    Args:
        velocity: [batch_dim, ...] shaped tensor; velocity model output
        x: [batch_dim, ...] shaped tensor; x_t data point
        t: [batch_dim,] time tensor
    """
    t = expand_t_like_x(t, xt)
    if path_type == "linear":
        alpha_t, d_alpha_t = 1 - t, torch.ones_like(xt, device=xt.device) * -1
        sigma_t, d_sigma_t = t, torch.ones_like(xt, device=xt.device)
    elif path_type == "cosine":
        alpha_t = torch.cos(t * np.pi / 2)
        sigma_t = torch.sin(t * np.pi / 2)
        d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
    else:
        raise NotImplementedError

    mean = xt
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * vt - mean) / var

    return score


def compute_diffusion(t_cur):
    return 2 * t_cur


def euler_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear", # not used, just for compatability
        collect_block_sim_at=None,  # list of 0-indexed step indices to collect block cosine similarity
        ):
    """
    Euler (ODE) sampler.

    If collect_block_sim_at is provided (list of step indices), at those steps the model is called
    with return_block_tokens=True and pairwise cosine similarity matrices are computed.
    When CFG is active, only the conditional branch tokens are used for analysis.

    Returns:
        x_next if collect_block_sim_at is None
        (x_next, all_sim_mats) if collect_block_sim_at is not None
            where all_sim_mats is a dict {step_idx: sim_mat [L, L] on CPU}
    """
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)
    _dtype = latents.dtype
    t_steps = torch.linspace(1, 0, num_steps+1, dtype=torch.float64)
    x_next = latents.to(torch.float64)
    device = x_next.device

    collect_set = set(collect_block_sim_at) if collect_block_sim_at is not None else set()
    all_sim_mats = {}

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            using_cfg = cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low
            if using_cfg:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur

            # Optionally collect block tokens for cosine similarity analysis
            if i in collect_set:
                out = model(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype),
                            return_block_tokens=True, **kwargs)
                d_cur = out[0].to(torch.float64)
                block_tokens = out[2]
                # CFG doubles the batch: take only the conditional branch (first half)
                if using_cfg:
                    B = x_cur.shape[0]
                    block_tokens = [h[:B] for h in block_tokens]
                all_sim_mats[i] = compute_block_cosine_matrix(block_tokens)
            else:
                d_cur = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    )[0].to(torch.float64)

            if using_cfg:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)
            x_next = x_cur + (t_next - t_cur) * d_cur

            if heun and (i < num_steps - 1):
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    model_input = torch.cat([x_next] * 2)
                    y_cur = torch.cat([y, y_null], dim=0)
                else:
                    model_input = x_next
                    y_cur = y
                kwargs = dict(y=y_cur)
                time_input = torch.ones(model_input.size(0)).to(
                    device=model_input.device, dtype=torch.float64
                    ) * t_next
                d_prime = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    )[0].to(torch.float64)
                if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
                    d_prime_cond, d_prime_uncond = d_prime.chunk(2)
                    d_prime = d_prime_uncond + cfg_scale * (d_prime_cond - d_prime_uncond)
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    if collect_block_sim_at is not None:
        return x_next, all_sim_mats
    return x_next


def euler_maruyama_sampler(
        model,
        latents,
        y,
        num_steps=20,
        heun=False,  # not used, just for compatability
        cfg_scale=1.0,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type="linear",
        collect_block_sim_at=None,  # list of 0-indexed step indices to collect block cosine similarity
        ):
    """
    Euler-Maruyama (SDE) sampler.

    If collect_block_sim_at is provided (list of step indices), at those steps the model is called
    with return_block_tokens=True and pairwise cosine similarity matrices are computed.
    When CFG is active, only the conditional branch tokens are used for analysis.

    Returns:
        mean_x if collect_block_sim_at is None
        (mean_x, all_sim_mats) if collect_block_sim_at is not None
            where all_sim_mats is a dict {step_idx: sim_mat [L, L] on CPU}
    """
    # setup conditioning
    if cfg_scale > 1.0:
        y_null = torch.tensor([1000] * y.size(0), device=y.device)

    _dtype = latents.dtype

    t_steps = torch.linspace(1., 0.04, num_steps, dtype=torch.float64)
    t_steps = torch.cat([t_steps, torch.tensor([0.], dtype=torch.float64)])
    x_next = latents.to(torch.float64)
    device = x_next.device

    collect_set = set(collect_block_sim_at) if collect_block_sim_at is not None else set()
    all_sim_mats = {}

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-2], t_steps[1:-1])):
            dt = t_next - t_cur
            x_cur = x_next
            using_cfg = cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low
            if using_cfg:
                model_input = torch.cat([x_cur] * 2, dim=0)
                y_cur = torch.cat([y, y_null], dim=0)
            else:
                model_input = x_cur
                y_cur = y
            kwargs = dict(y=y_cur)
            time_input = torch.ones(model_input.size(0)).to(device=device, dtype=torch.float64) * t_cur
            diffusion = compute_diffusion(t_cur)
            eps_i = torch.randn_like(x_cur).to(device)
            deps = eps_i * torch.sqrt(torch.abs(dt))

            # Optionally collect block tokens for cosine similarity analysis
            if i in collect_set:
                out = model(model_input.to(dtype=_dtype), time_input.to(dtype=_dtype),
                            return_block_tokens=True, **kwargs)
                v_cur = out[0].to(torch.float64)
                block_tokens = out[2]
                if using_cfg:
                    B = x_cur.shape[0]
                    block_tokens = [h[:B] for h in block_tokens]
                all_sim_mats[i] = compute_block_cosine_matrix(block_tokens)
            else:
                v_cur = model(
                    model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
                    )[0].to(torch.float64)

            # compute drift
            s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
            d_cur = v_cur - 0.5 * diffusion * s_cur
            if using_cfg:
                d_cur_cond, d_cur_uncond = d_cur.chunk(2)
                d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

            x_next = x_cur + d_cur * dt + torch.sqrt(diffusion) * deps

    # last step
    t_cur, t_next = t_steps[-2], t_steps[-1]
    dt = t_next - t_cur
    x_cur = x_next
    if cfg_scale > 1.0 and t_cur <= guidance_high and t_cur >= guidance_low:
        model_input = torch.cat([x_cur] * 2, dim=0)
        y_cur = torch.cat([y, y_null], dim=0)
    else:
        model_input = x_cur
        y_cur = y
    kwargs = dict(y=y_cur)
    time_input = torch.ones(model_input.size(0)).to(
        device=device, dtype=torch.float64
        ) * t_cur

    # compute drift
    v_cur = model(
        model_input.to(dtype=_dtype), time_input.to(dtype=_dtype), **kwargs
        )[0].to(torch.float64)
    s_cur = get_score_from_velocity(v_cur, model_input, time_input, path_type=path_type)
    diffusion = compute_diffusion(t_cur)
    d_cur = v_cur - 0.5 * diffusion * s_cur
    if cfg_scale > 1. and t_cur <= guidance_high and t_cur >= guidance_low:
        d_cur_cond, d_cur_uncond = d_cur.chunk(2)
        d_cur = d_cur_uncond + cfg_scale * (d_cur_cond - d_cur_uncond)

    mean_x = x_cur + dt * d_cur

    if collect_block_sim_at is not None:
        return mean_x, all_sim_mats
    return mean_x
