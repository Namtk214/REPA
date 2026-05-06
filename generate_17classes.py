#!/usr/bin/env python3
"""
Generate 17 classes, each with 12 images, CFG=4.0
Modified from generate.py
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import argparse
from samplers import euler_sampler, euler_maruyama_sampler
from utils import load_legacy_checkpoints, download_model


def main(args):
    """
    Run sampling for specific classes.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling requires at least one GPU"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Randomly select 17 classes from 1000 ImageNet classes
    # Use the seed for reproducibility
    torch.manual_seed(args.global_seed)
    num_target_classes = 5
    target_classes = torch.randperm(args.num_classes)[:num_target_classes].tolist()
    target_classes.sort()  # Sort for easier tracking

    images_per_class = 12  # 12 images per class

    if rank == 0:
        print(f"\nGenerating {len(target_classes)} classes x {images_per_class} images/class")
        print(f"Total images: {len(target_classes) * images_per_class}")
        print(f"Classes: {target_classes}\n")

    # Load model
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=True,
        z_dims=[int(z_dim) for z_dim in args.projector_embed_dims.split(',')],
        encoder_depth=args.encoder_depth,
        **block_kwargs,
    ).to(device)

    # Load checkpoint
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
    model.eval()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    if rank == 0:
        print(f"Model loaded successfully!")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Projector Parameters: {sum(p.numel() for p in model.projectors.parameters()):,}")

    # Create output folder
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-17classes-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"\nSaving images to: {sample_folder_dir}\n")
    dist.barrier()

    # Generate images for each class
    world_size = dist.get_world_size()

    for class_idx in tqdm(target_classes, desc="Classes", disable=(rank != 0)):
        # Create class folder
        class_folder = f"{sample_folder_dir}/class_{class_idx:04d}"
        if rank == 0:
            os.makedirs(class_folder, exist_ok=True)
        dist.barrier()

        # Calculate how many images each GPU should generate
        images_per_gpu = images_per_class // world_size
        remainder = images_per_class % world_size

        # Distribute remainder images to first few GPUs
        if rank < remainder:
            my_images = images_per_gpu + 1
            offset = rank * (images_per_gpu + 1)
        else:
            my_images = images_per_gpu
            offset = remainder * (images_per_gpu + 1) + (rank - remainder) * images_per_gpu

        if my_images == 0:
            continue

        # Generate images in batches
        batch_size = args.per_proc_batch_size
        num_batches = (my_images + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            # Calculate batch size for this iteration
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, my_images)
            current_batch_size = end_idx - start_idx

            # Sample latents
            z = torch.randn(current_batch_size, model.in_channels, latent_size, latent_size, device=device)
            y = torch.full((current_batch_size,), class_idx, device=device, dtype=torch.long)

            # Sample images
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
            )

            with torch.no_grad():
                if args.mode == "sde":
                    samples = euler_maruyama_sampler(**sampling_kwargs)
                elif args.mode == "ode":
                    samples = euler_sampler(**sampling_kwargs)
                else:
                    raise NotImplementedError()

                if isinstance(samples, tuple):
                    samples = samples[0]

                samples = samples.to(torch.float32)

                # Decode latents
                latents_scale = torch.tensor([0.18215] * 4).view(1, 4, 1, 1).to(device)
                latents_bias = -torch.tensor([0.] * 4).view(1, 4, 1, 1).to(device)
                samples = vae.decode((samples - latents_bias) / latents_scale).sample
                samples = (samples + 1) / 2.0
                samples = torch.clamp(255.0 * samples, 0, 255)
                samples = samples.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

                # Save images
                for i, sample in enumerate(samples):
                    img_idx = offset + start_idx + i
                    filename = f"{class_folder}/{img_idx:04d}.png"
                    Image.fromarray(sample).save(filename)

        dist.barrier()

    dist.barrier()
    if rank == 0:
        print(f"\nDone! Generated images saved to: {sample_folder_dir}")
        print(f"Total: {len(target_classes)} classes x {images_per_class} images = {len(target_classes) * images_per_class} images")

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--global-seed", type=int, default=42)

    # precision
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)

    # logging/saving
    parser.add_argument("--ckpt", type=str, default=None, help="Optional path to checkpoint")
    parser.add_argument("--sample-dir", type=str, default="samples")

    # model
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # vae
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")

    # batch size
    parser.add_argument("--per-proc-batch-size", type=int, default=4)

    # sampling parameters - CFG=4.0 by default
    parser.add_argument("--mode", type=str, default="sde", choices=["sde", "ode"])
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--projector-embed-dims", type=str, default="768")
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=0.7)

    # legacy
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    main(args)
