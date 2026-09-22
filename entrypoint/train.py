"""
Kaggle 2x NVIDIA T4 Distributed (DDP) Entrypoint Training Script.
Supports 4 Fairly Benchmarked MoE Models:
1. 'erc': MoE with ERC Loss (ICLR 2026)
2. 'vanilla': Vanilla MoE (Switch Transformer)
3. 'aoe': Autonomy-of-Experts (Lv et al., ICML 2025)
4. 'deepseek': DeepSeek-MoE with Shared + Routed Experts (Dai et al., 2024)
"""

import os
import sys

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils import (
    configure_adamw_optimizer,
    get_cosine_schedule_with_min_lr,
    get_dolma_urls,
    ShardedStreamingDataset
)
from models import (
    create_fair_moe_config,
    build_erc_moe_model,
    build_vanilla_moe_model,
    build_aoe_model,
    build_deepseek_moe_model
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fair MoE Benchmark on Kaggle 2x T4")
    parser.add_argument("--model_type", type=str, default="erc", choices=["erc", "vanilla", "aoe", "deepseek"],
                        help="Model to train: 'erc', 'vanilla', 'aoe', or 'deepseek'")
    parser.add_argument("--dolma_num_shards", type=int, default=30, help="Number of Dolma shards to stream")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (context window)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU (effective global batch = 64)")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=4e-4, help="Peak learning rate (paper uses 4e-4)")
    parser.add_argument("--min_lr", type=float, default=4e-5, help="Minimum cosine learning rate (paper uses 4e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (paper uses 0.1)")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1 (paper uses 0.9)")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2 (paper uses 0.95)")
    parser.add_argument("--max_steps", type=int, default=4000, help="Max steps (4000 steps ~ 131M tokens, ~3.5h on 2x T4)")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--alpha", type=float, default=1.0, help="ERC threshold parameter alpha (default 1.0)")
    parser.add_argument("--erc_weight", type=float, default=1.0, help="ERC loss weight (default 1.0)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/checkpoints", help="Output directory")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging to Weights & Biases (wandb.ai)")
    parser.add_argument("--wandb_project", type=str, default="moe-erc-iclr2026", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="", help="WandB run name")
    return parser.parse_args()


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    args = parse_args()
    rank, local_rank, world_size = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"=== Starting Training | Model: {args.model_type.upper()} | World Size: {world_size} GPUs ===", flush=True)
        if args.use_wandb:
            try:
                import wandb
                default_name = f"{args.model_type}_moe_110m"
                run_name = args.wandb_run_name if args.wandb_run_name else default_name
                wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
                print(f"WandB initialized: project={args.wandb_project}, run={run_name}", flush=True)
            except Exception as e:
                print(f"[Warning] Failed to initialize wandb: {e}", flush=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # 1. Initialize Tokenizer & Base Config
    if is_main:
        print("Loading tokenizer allenai/OLMoE-1B-7B-0924...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = create_fair_moe_config(vocab_size=len(tokenizer))

    # 2. Build Selected Architecture (Fair 110M / Active ~35M parameters)
    if is_main:
        print(f"Building model architecture: {args.model_type}...", flush=True)
    if args.model_type == "erc":
        model = build_erc_moe_model(config, alpha=args.alpha, erc_weight=args.erc_weight, record_matrices=is_main)
    elif args.model_type == "vanilla":
        model = build_vanilla_moe_model(config)
    elif args.model_type == "aoe":
        model = build_aoe_model(config)
    elif args.model_type == "deepseek":
        model = build_deepseek_moe_model(config)

    model.to(device)
    if is_main:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total model parameters: {num_params / 1e6:.2f} M", flush=True)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 3. Dataset Loader (Streaming Dolma v1.5 sample)
    if is_main:
        print(f"Fetching Dolma v1.5 shard URLs ({args.dolma_num_shards} shards)...", flush=True)
    urls = get_dolma_urls(args.dolma_num_shards)
    dataset = ShardedStreamingDataset(urls, tokenizer, seq_len=args.seq_len, rank=rank, world_size=world_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)
    if is_main:
        print("Dataloader ready. Configuring optimizer...", flush=True)

    # 4. AdamW Optimizer & Cosine Schedule (Section 4.1)
    raw_model = model.module if hasattr(model, "module") else model
    optimizer = configure_adamw_optimizer(
        raw_model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    scheduler = get_cosine_schedule_with_min_lr(
        optimizer,
        num_warmup_steps=int(args.max_steps * 0.05),
        num_training_steps=args.max_steps,
        peak_lr=args.lr,
        min_lr=args.min_lr
    )
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    start_step = 0
    history = []

    def save_checkpoint(curr_step: int, tag: str = "latest"):
        """Saves full training state for resume and evaluation."""
        if not is_main:
            return
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_{tag}.pt")
        print(f"--> [Save Checkpoint] Step {curr_step} -> {ckpt_path}...", flush=True)
        torch.save({
            "step": curr_step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "model_type": args.model_type,
            "config": config.to_dict()
        }, ckpt_path)

    # Auto-resume logic
    resume_target = args.resume
    if not resume_target or resume_target.lower() == "auto":
        default_ckpt = os.path.join(args.output_dir, "checkpoint_latest.pt")
        if os.path.exists(default_ckpt):
            resume_target = default_ckpt

    if resume_target and os.path.exists(resume_target):
        if is_main:
            print(f"--> [Resume] Loading checkpoint from: {resume_target}", flush=True)
        ckpt = torch.load(resume_target, map_location=device)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        history = ckpt.get("history", [])
        if is_main:
            print(f"--> [Resume] Successfully resumed from step {start_step}!", flush=True)

    # Signal handlers for graceful Kaggle timeout / cancellation
    current_training_step = [start_step]

    def handle_interrupt(signum, frame):
        if is_main:
            print(f"\n[Signal {signum} received!] Emergency saving checkpoint at step {current_training_step[0]}...", flush=True)
            save_checkpoint(current_training_step[0], tag="latest")
            with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)
        cleanup_distributed()
        sys.exit(0)

    try:
        import signal
        signal.signal(signal.SIGINT, handle_interrupt)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, handle_interrupt)
    except Exception:
        pass

    # 5. Training Loop
    raw_model.train()
    step = start_step
    accum_loss = 0.0
    start_time = time.time()
    data_iter = iter(dataloader)

    if is_main:
        print(f"Training from step {step} to {args.max_steps}...", flush=True)

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        while step < args.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            step += 1
            current_training_step[0] = step
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type, dtype=torch.float16):
                outputs = model(input_ids=input_ids, labels=labels)
                task_loss = outputs.loss
                if args.model_type in ["aoe", "deepseek"]:
                    aux_loss = sum(l.mlp.last_aux_loss for l in raw_model.model.layers) / len(raw_model.model.layers)
                    total_loss = task_loss + aux_loss
                else:
                    total_loss = task_loss
                loss = total_loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if step % args.grad_accum_steps == 0 or step == args.max_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                effective_accum = (step % args.grad_accum_steps) if (step % args.grad_accum_steps) != 0 else args.grad_accum_steps
                if is_main and ((step // args.grad_accum_steps) % 10 == 0 or step == args.max_steps):
                    current_loss = (accum_loss / effective_accum) * args.grad_accum_steps
                    erc_val = getattr(raw_model, "_last_erc_loss", torch.tensor(0.0)).item()
                    lr_val = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    tok_speed = (step * args.batch_size * args.seq_len * world_size) / (elapsed + 1e-6)

                    print(f"Step {step:4d}/{args.max_steps} | Loss: {current_loss:.4f} | "
                          f"ERC Loss: {erc_val:.4f} | LR: {lr_val:.2e} | Speed: {tok_speed:.0f} tok/s", flush=True)
                    
                    history.append({
                        "step": step,
                        "loss": round(current_loss, 4),
                        "erc_loss": round(erc_val, 4),
                        "lr": lr_val
                    })
                    with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                        json.dump(history, f, indent=2)

                    if args.use_wandb:
                        try:
                            import wandb
                            wandb_payload = {
                                "train/total_loss": current_loss,
                                "train/erc_loss": erc_val,
                                "train/learning_rate": lr_val,
                                "train/throughput_tokens_per_sec": tok_speed,
                                "train/tokens_trained": step * args.batch_size * args.seq_len * world_size,
                            }
                            if hasattr(raw_model, "_last_erc_matrices") and raw_model._last_erc_matrices:
                                M = raw_model._last_erc_matrices[0]
                                n = M.size(0)
                                diag = torch.diag(M).mean().item()
                                mask = torch.ones_like(M) - torch.eye(n, device=M.device)
                                off_diag = (M * mask).sum().item() / (n * (n - 1) + 1e-8)
                                wandb_payload["train/specialization_ratio"] = diag / (off_diag + 1e-8)
                            wandb.log(wandb_payload, step=step)
                        except Exception:
                            pass

                accum_loss = 0.0

            # Periodic checkpoint saving
            if is_main and (step % args.save_every == 0 or step == args.max_steps):
                save_checkpoint(step, tag="latest")

        # 6. Final Model Saving
        if is_main:
            print("\n=== Training Completed: Saving Final Model & Artifacts ===", flush=True)
            # Save checkpoint_final.pt and standalone weights
            save_checkpoint(step, tag="final")
            weights_path = os.path.join(args.output_dir, "model_final.pt")
            torch.save(raw_model.state_dict(), weights_path)
            print(f"--> Saved standalone model weights to {weights_path}", flush=True)

            # Save HF transformers artifacts
            try:
                raw_model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"--> Saved Hugging Face model and tokenizer to {args.output_dir}", flush=True)
            except Exception as e:
                print(f"[Warning] save_pretrained failed: {e}", flush=True)

            with open(os.path.join(args.output_dir, "training_history.json"), "w") as f:
                json.dump(history, f, indent=2)
            print("Done training successfully!", flush=True)

    finally:
        if is_main and args.use_wandb:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        cleanup_distributed()


if __name__ == "__main__":
    main()
