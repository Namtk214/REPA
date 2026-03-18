# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

Block-wise cosine similarity visualization:
  Add --track-similarity to collect [L, L] cosine similarity matrices at
  specific noise levels (t values) and save visualizations as PNG files.
  Tracking runs once on rank-0's first batch only (zero overhead elsewhere).
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def noise_levels_to_step_indices(noise_levels, num_steps, mode):
    """
    Convert continuous noise level t values to 0-indexed sampler step indices.

    For ODE (euler_sampler):
        t_steps = linspace(1, 0, num_steps+1)  → step i starts at t=t_steps[i]

    For SDE (euler_maruyama_sampler):
        t_steps = linspace(1.0, 0.04, num_steps)  → step i starts at t=t_steps[i]
        (plus one final step to t=0)

    Returns:
        step_idx_to_t: dict {step_idx (int): t_value (float)}
                       mapping each chosen step to its target noise level.
    """
    if mode == "ode":
        t_ref = torch.linspace(1.0, 0.0, num_steps + 1)[:-1]  # [num_steps] starts of each step
    else:  # sde
        t_ref = torch.linspace(1.0, 0.04, num_steps)           # [num_steps]

    step_idx_to_t = {}
    for t_target in noise_levels:
        diffs = torch.abs(t_ref - t_target)
        idx = int(diffs.argmin())
        if idx not in step_idx_to_t:  # keep first match if two levels map to same step
            step_idx_to_t[idx] = round(t_target, 4)

    return step_idx_to_t


def save_block_sim_visualizations(sim_mats_by_t, out_dir, args):
    """
    Save block cosine similarity visualizations keyed by noise level (t value).

    Outputs:
      <out_dir>/block_sim_t{t:.1f}.png  — individual heatmap per noise level
      <out_dir>/block_sim_grid.png      — all noise levels in one figure
      <out_dir>/block_sim_avg.png       — average matrix across noise levels
      <out_dir>/block_sim_data.npz      — raw numpy arrays

    Args:
        sim_mats_by_t: dict {t_value (float): numpy [L, L]}
        out_dir: directory to write files into
        args: parsed argparse namespace (used for plot titles)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    sorted_items = sorted(sim_mats_by_t.items(), key=lambda x: -x[0])  # high t (noisy) first
    L = next(iter(sim_mats_by_t.values())).shape[0]
    block_ticks = list(range(L))

    def _draw_heatmap(ax, mat, title):
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(block_ticks)
        ax.set_yticks(block_ticks)
        ax.set_xticklabels(block_ticks, fontsize=7)
        ax.set_yticklabels(block_ticks, fontsize=7)
        ax.set_xlabel("Block index", fontsize=8)
        ax.set_ylabel("Block index", fontsize=8)
        ax.set_title(title, fontsize=8)
        return im

    # ── 1. Individual heatmap per noise level ─────────────────────────────────
    for t_val, sim_mat in sorted_items:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = _draw_heatmap(ax, sim_mat,
                           f"Block Cosine Sim  |  t = {t_val:.2f}  |  cfg = {args.cfg_scale}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        path = os.path.join(out_dir, f"block_sim_t{t_val:.2f}.png")
        plt.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  Saved {path}")

    # ── 2. Grid: all noise levels in one figure ────────────────────────────────
    n_plots = len(sorted_items)
    ncols = min(n_plots, 5)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes_flat = np.array(axes).flatten()
    for ax_i, (t_val, sim_mat) in enumerate(sorted_items):
        im = _draw_heatmap(axes_flat[ax_i], sim_mat, f"t = {t_val:.2f}")
        plt.colorbar(im, ax=axes_flat[ax_i], fraction=0.046, pad=0.04)
    for ax_i in range(n_plots, len(axes_flat)):
        axes_flat[ax_i].set_visible(False)
    fig.suptitle(
        f"Block-wise Cosine Similarity across Noise Levels\n"
        f"model={args.model}  mode={args.mode}  cfg={args.cfg_scale}",
        fontsize=9,
    )
    plt.tight_layout()
    grid_path = os.path.join(out_dir, "block_sim_grid.png")
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {grid_path}")

    # ── 3. Average across noise levels ────────────────────────────────────────
    avg_mat = np.mean(list(sim_mats_by_t.values()), axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = _draw_heatmap(ax, avg_mat,
                       f"Block Cosine Sim  (avg over t)  |  cfg = {args.cfg_scale}")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    avg_path = os.path.join(out_dir, "block_sim_avg.png")
    plt.savefig(avg_path, dpi=120)
    plt.close(fig)
    print(f"  Saved {avg_path}")

    # ── 4. Raw matrices ────────────────────────────────────────────────────────
    npz_path = os.path.join(out_dir, "block_sim_data.npz")
    save_dict = {f"t{t_val:.2f}".replace(".", "_"): v for t_val, v in sim_mats_by_t.items()}
    save_dict["average"] = avg_mat
    np.savez(npz_path, **save_dict)
    print(f"  Saved raw matrices → {npz_path}")
    print(f"\n[block_sim] Done. {n_plots} noise levels visualized in {out_dir}")


def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:cd
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Load model:
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = True,
        z_dims = [int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt
    if ckpt_path is None:
        args.ckpt = 'SiT-XL-2-256x256.pt'
        assert args.model == 'SiT-XL/2'
        assert len(args.projector_embed_dims.split(',')) == 1
        assert int(args.projector_embed_dims.split(',')[0]) == 768
        state_dict = download_model('last.pt')
    else:
        state_dict = torch.load(ckpt_path, map_location=f'cuda:{device}')['ema']
    if args.legacy:
        state_dict = load_legacy_checkpoints(
            state_dict=state_dict, encoder_depth=args.encoder_depth
            )
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"

    # Pre-compute step indices for block similarity tracking (rank 0 only)
    step_idx_to_t = {}
    if args.track_similarity and rank == 0:
        step_idx_to_t = noise_levels_to_step_indices(
            args.track_noise_levels, args.num_steps, args.mode
        )
        print(f"[block_sim] Tracking noise levels: {args.track_noise_levels}")
        print(f"[block_sim] Mapped to step indices: {sorted(step_idx_to_t.keys())}")

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.resolution}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.mode}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    sim_tracked = False  # track block similarity only once (first batch, rank 0)

    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Collect block similarity on the very first batch of rank 0 only (zero overhead afterwards)
        do_track = args.track_similarity and rank == 0 and not sim_tracked
        collect_at = sorted(step_idx_to_t.keys()) if do_track else None

        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
            collect_block_sim_at=collect_at,
        )
        with torch.no_grad():
            if args.mode == "sde":
                result = euler_maruyama_sampler(**sampling_kwargs)
            elif args.mode == "ode":
                result = euler_sampler(**sampling_kwargs)
            else:
                raise NotImplementedError()

            if do_track and isinstance(result, tuple):
                samples_lat, all_sim_mats = result
                # Remap step_idx → t_value for human-readable visualization
                sim_mats_by_t = {
                    step_idx_to_t[idx]: mat.numpy()
                    for idx, mat in all_sim_mats.items()
                    if idx in step_idx_to_t
                }
                sim_out_dir = args.sim_out_dir or os.path.join(sample_folder_dir, "block_sim")
                print(f"\n[block_sim] Saving visualizations to {sim_out_dir}/")
                save_block_sim_visualizations(sim_mats_by_t, sim_out_dir, args)
                sim_tracked = True
            else:
                samples_lat = result if not isinstance(result, tuple) else result[0]

            samples_lat = samples_lat.to(torch.float32)

            latents_scale = torch.tensor(
                [0.18215, 0.18215, 0.18215, 0.18215, ]
                ).view(1, 4, 1, 1).to(device)
            latents_bias = -torch.tensor(
                [0., 0., 0., 0.,]
                ).view(1, 4, 1, 1).to(device)
            samples = vae.decode((samples_lat - latents_bias) / latents_scale).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(
                255. * samples, 0, 255
                ).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + rank + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=0)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")

    # logging/saving:
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to a SiT checkpoint.")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")

    # number of samples
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)

    # sampling related hyperparameters
    parser.add_argument("--mode", type=str, default="ode")
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--projector-embed-dims", type=str, default="768,1024")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False) # only for ode
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    # will be deprecated
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False) # only for ode

    # block-wise cosine similarity visualization
    parser.add_argument("--track-similarity", action="store_true",
                        help="Collect block-wise cosine similarity at specified noise levels and "
                             "save PNG visualizations. Runs once on rank-0 first batch only.")
    parser.add_argument("--track-noise-levels", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="Noise levels (t values) to track, e.g. 0.1 0.2 ... 1.0. "
                             "Each is mapped to the closest sampler step for the chosen mode/num-steps.")
    parser.add_argument("--sim-out-dir", type=str, default=None,
                        help="Directory to save similarity PNGs and .npz. "
                             "Defaults to <sample-dir>/<run-name>/block_sim/")

    args = parser.parse_args()
    main(args)
