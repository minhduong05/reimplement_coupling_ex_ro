"""
Comprehensive Evaluation Suite for MoE Architectures (ICLR 2026 Reimplementation)
Evaluates downstream benchmarks from the paper:
- Language Modeling Perplexity: Wikitext-2
- Multiple-Choice Reasoning & Knowledge:
    * HellaSwag (Zellers et al., 2019)
    * ARC-Challenge (Clark et al., 2018)
    * SciQ (Welbl et al., 2017)
    * OpenBookQA (Mihaylov et al., 2018)
    * BoolQ (Clark et al., 2019)
    * WinoGrande (Sakaguchi et al., 2021)
    * MMLU (Hendrycks et al., 2021)
- Efficiency & Latency: Tokens per second (Throughput) & Latency (ms/token)
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# Adjust path to import models and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.config import create_fair_moe_config
from models.moe_erc import build_erc_moe_model
from models.moe_vanilla import build_vanilla_moe_model
from models.aoe import build_aoe_model
from models.deepseek_moe import build_deepseek_moe_model


# ==============================================================================
# 1. Model Loading
# ==============================================================================

def find_checkpoint_path(target_path_or_dir: str, model_type: str) -> Optional[str]:
    """
    Intelligently discovers checkpoint file from:
    1. Direct file path
    2. Subdirectory named checkpoints_{model_type} or {model_type}
    3. Kaggle input/working directories recursively
    """
    # Direct file
    if os.path.isfile(target_path_or_dir):
        return target_path_or_dir
        
    if not os.path.exists(target_path_or_dir):
        return None

    # Common subfolder patterns
    candidate_dirs = [
        target_path_or_dir,
        os.path.join(target_path_or_dir, f"checkpoints_{model_type}"),
        os.path.join(target_path_or_dir, model_type),
        os.path.join(target_path_or_dir, f"{model_type}_moe_110m"),
    ]
    
    # Also check /kaggle/input and /kaggle/working if available
    for extra_base in ["/kaggle/working", "/kaggle/input"]:
        if os.path.exists(extra_base):
            candidate_dirs.append(os.path.join(extra_base, f"checkpoints_{model_type}"))
            candidate_dirs.append(os.path.join(extra_base, model_type))
            # Scan one level of subfolders in /kaggle/input
            try:
                for sub in os.listdir(extra_base):
                    sub_path = os.path.join(extra_base, sub)
                    if os.path.isdir(sub_path):
                        candidate_dirs.append(os.path.join(sub_path, f"checkpoints_{model_type}"))
                        candidate_dirs.append(os.path.join(sub_path, model_type))
            except Exception:
                pass

    candidate_files = [
        "checkpoint_latest.pt",
        "checkpoint_step_4000.pt",
        "checkpoint_step_3000.pt",
        "checkpoint_step_2000.pt",
        "model.pt",
        "pytorch_model.pt",
        "checkpoint.pt"
    ]

    for d in candidate_dirs:
        if os.path.isdir(d):
            # Check explicit filenames first
            for f in candidate_files:
                p = os.path.join(d, f)
                if os.path.isfile(p):
                    return p
            # Check any .pt file containing model_type or latest
            pts = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pt")]
            if pts:
                return pts[0]

    # Recursive walk if still not found
    for root, _, files in os.walk(target_path_or_dir):
        for f in files:
            if f.endswith(".pt") and (model_type in root.lower() or model_type in f.lower()):
                return os.path.join(root, f)

    return None


def load_moe_model(model_type: str, ckpt_dir: str, tokenizer: AutoTokenizer, device: torch.device):
    """Loads a specific MoE model from its checkpoint directory or file."""
    config = create_fair_moe_config(vocab_size=len(tokenizer))
    
    if model_type == "erc":
        model = build_erc_moe_model(config)
    elif model_type == "vanilla":
        model = build_vanilla_moe_model(config)
    elif model_type == "aoe":
        model = build_aoe_model(config)
    elif model_type == "deepseek":
        model = build_deepseek_moe_model(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    ckpt_path = find_checkpoint_path(ckpt_dir, model_type)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found for '{model_type}' in directory or candidates from: {ckpt_dir}")

    print(f"--> Loading {model_type.upper()} weights from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    
    # Strip potential 'module.' prefix if DDP was saved directly
    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model.to(device)
    model.eval()
    return model


# ==============================================================================
# 2. Evaluation Helpers: Conditional Log-Likelihood
# ==============================================================================

@torch.no_grad()
def get_log_likelihood(model, tokenizer, prompt: str, continuation: str, device: torch.device) -> float:
    """Computes length-normalized log P(continuation | prompt)."""
    full_text = prompt + continuation
    full_tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    prompt_len = prompt_tokens.shape[1]
    full_len = full_tokens.shape[1]
    
    if full_len <= prompt_len:
        return -9999.0

    outputs = model(input_ids=full_tokens)
    logits = outputs.logits  # (1, seq_len, vocab_size)
    
    # Target tokens start at index prompt_len up to full_len
    # Prediction for token at position t is from logits at position t - 1
    shift_logits = logits[0, prompt_len - 1 : full_len - 1, :]
    shift_labels = full_tokens[0, prompt_len : full_len]
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # Return average log-likelihood per token (length-normalized)
    return target_log_probs.sum().item() / max(1, (full_len - prompt_len))


# ==============================================================================
# 3. Individual Benchmark Evaluators
# ==============================================================================

def eval_wikitext2(model, tokenizer, device: torch.device, max_tokens: int = 50000) -> Dict[str, float]:
    """Calculates Language Modeling Loss and Perplexity on Wikitext-2 test split."""
    from datasets import load_dataset
    print("  [Task] Evaluating Wikitext-2 Perplexity...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join([x["text"] for x in dataset if x["text"].strip()])
    
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = 512
    stride = 256
    
    total_tokens = min(encodings.input_ids.size(1), max_tokens)
    input_ids = encodings.input_ids[:, :total_tokens].to(device)
    
    nlls = []
    with torch.no_grad():
        for i in range(0, input_ids.size(1) - 1, stride):
            begin_loc = max(i + stride - seq_len, 0)
            end_loc = min(i + stride, input_ids.size(1))
            trg_len = end_loc - i
            
            chunk_input = input_ids[:, begin_loc:end_loc]
            target_ids = chunk_input.clone()
            target_ids[:, :-trg_len] = -100
            
            outputs = model(chunk_input, labels=target_ids)
            neg_log_likelihood = outputs.loss
            nlls.append(neg_log_likelihood.item())
            
            if end_loc == input_ids.size(1):
                break
                
    mean_loss = float(np.mean(nlls))
    ppl = math.exp(mean_loss)
    return {"wikitext_loss": round(mean_loss, 4), "wikitext_ppl": round(ppl, 2)}


def eval_hellaswag(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates HellaSwag zero-shot accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating HellaSwag ({max_samples} samples)...")
    dataset = load_dataset("hellaswag", split="validation")
    
    correct = 0
    total = min(len(dataset), max_samples)
    for idx in range(total):
        item = dataset[idx]
        ctx = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        
        scores = [get_log_likelihood(model, tokenizer, ctx + " ", ending, device) for ending in endings]
        pred = int(np.argmax(scores))
        if pred == label:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_arc_challenge(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates ARC-Challenge accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating ARC-Challenge ({max_samples} samples)...")
    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    
    correct = 0
    total = min(len(dataset), max_samples)
    for idx in range(total):
        item = dataset[idx]
        q = f"Question: {item['question']}\nAnswer:"
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold_key = item["answerKey"]
        
        # Find index of gold label
        gold_idx = labels.index(gold_key) if gold_key in labels else 0
        scores = [get_log_likelihood(model, tokenizer, q + " ", c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_sciq(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates SciQ science exam accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating SciQ ({max_samples} samples)...")
    dataset = load_dataset("sciq", split="test")
    
    correct = 0
    total = min(len(dataset), max_samples)
    for idx in range(total):
        item = dataset[idx]
        q = f"Question: {item['question']}\nAnswer:"
        # SciQ has 3 distractors + 1 correct answer
        choices = [item["distractor1"], item["distractor2"], item["distractor3"], item["correct_answer"]]
        gold_idx = 3 # correct answer is always at index 3 in this list
        
        scores = [get_log_likelihood(model, tokenizer, q + " ", c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_openbookqa(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates OpenBookQA accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating OpenBookQA ({max_samples} samples)...")
    dataset = load_dataset("openbookqa", "main", split="test")
    
    correct = 0
    total = min(len(dataset), max_samples)
    for idx in range(total):
        item = dataset[idx]
        q = f"Question: {item['question_stem']}\nAnswer:"
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold_key = item["answerKey"]
        
        gold_idx = labels.index(gold_key) if gold_key in labels else 0
        scores = [get_log_likelihood(model, tokenizer, q + " ", c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_boolq(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates BoolQ (yes/no reading comprehension) accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating BoolQ ({max_samples} samples)...")
    dataset = load_dataset("google/boolq", split="validation")
    
    correct = 0
    total = min(len(dataset), max_samples)
    choices = ["false", "true"]
    for idx in range(total):
        item = dataset[idx]
        prompt = f"Passage: {item['passage']}\nQuestion: {item['question']}?\nAnswer:"
        gold_idx = 1 if item["answer"] else 0
        
        scores = [get_log_likelihood(model, tokenizer, prompt + " ", c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_winogrande(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates WinoGrande common sense pronoun resolution."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating WinoGrande ({max_samples} samples)...")
    dataset = load_dataset("winogrande", "winogrande_xs", split="validation")
    
    correct = 0
    total = min(len(dataset), max_samples)
    for idx in range(total):
        item = dataset[idx]
        sent = item["sentence"]
        # Split at underscore
        parts = sent.split("_")
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        
        c1 = item["option1"] + suffix
        c2 = item["option2"] + suffix
        choices = [c1, c2]
        gold_idx = int(item["answer"]) - 1  # answer is '1' or '2'
        
        scores = [get_log_likelihood(model, tokenizer, prefix, c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_mmlu(model, tokenizer, device: torch.device, max_samples: int = 200) -> float:
    """Evaluates MMLU subset accuracy."""
def eval_commonsenseqa(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates CommonsenseQA (Talmor et al., 2019) accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating CommonsenseQA ({max_samples} samples)...")
    try:
        dataset = load_dataset("tau/commonsense_qa", split="validation")
    except Exception:
        dataset = load_dataset("commonsense_qa", split="validation")
        
    correct = 0
    total = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    for idx in range(total):
        item = dataset[idx]
        q = f"Question: {item['question']}\nAnswer:"
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        gold_key = item["answerKey"]
        gold_idx = labels.index(gold_key) if gold_key in labels else 0
        
        scores = [get_log_likelihood(model, tokenizer, q + " ", c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_copa(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates COPA - Choice of Plausible Alternatives (Roemmele et al., 2011)."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating COPA ({max_samples} samples)...")
    try:
        dataset = load_dataset("aps/super_glue", "copa", split="validation")
    except Exception:
        dataset = load_dataset("super_glue", "copa", split="validation")
        
    correct = 0
    total = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    for idx in range(total):
        item = dataset[idx]
        premise = item["premise"].rstrip(".")
        question = item["question"]  # 'cause' or 'effect'
        connector = " because " if question == "cause" else " so "
        prompt = premise + connector
        
        choices = [item["choice1"], item["choice2"]]
        gold_idx = int(item["label"])  # 0 or 1
        
        scores = [get_log_likelihood(model, tokenizer, prompt, c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_social_i_qa(model, tokenizer, device: torch.device, max_samples: int = 300) -> float:
    """Evaluates Social IQa (Sap et al., 2019) social commonsense accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating Social IQa ({max_samples} samples)...")
    try:
        dataset = load_dataset("allenai/social_i_qa", revision="refs/convert/parquet", split="validation")
    except Exception:
        try:
            dataset = load_dataset("allenai/social_i_qa", split="validation")
        except Exception:
            dataset = load_dataset("social_i_qa", split="validation")
            
    correct = 0
    total = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    for idx in range(total):
        item = dataset[idx]
        context = item["context"]
        question = item["question"]
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        
        choices = [item["answerA"], item["answerB"], item["answerC"]]
        gold_idx = int(item["label"]) - 1
        
        scores = [get_log_likelihood(model, tokenizer, prompt + " ", c, device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_gsm8k(model, tokenizer, device: torch.device, max_samples: int = 200) -> float:
    """Evaluates GSM8K grade school math reasoning (Cobbe et al., 2021)."""
    from datasets import load_dataset
    import re
    print(f"  [Task] Evaluating GSM8K ({max_samples} samples)...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    correct = 0
    total = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    for idx in range(total):
        item = dataset[idx]
        q = f"Question: {item['question']}\nLet's think step by step.\nAnswer: "
        gold_ans = item["answer"].split("####")[-1].strip().replace(",", "")
        
        # Greedy generate short answer
        inputs = tokenizer(q, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                inputs.input_ids,
                max_new_tokens=48,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            ) if hasattr(model, "generate") else None
            
        if out is not None:
            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", gen_text)
            pred_num = numbers[-1] if numbers else ""
            if pred_num == gold_ans:
                correct += 1

    acc = correct / max(1, total) * 100.0
    return round(acc, 2)


def eval_mmlu(model, tokenizer, device: torch.device, max_samples: int = 200) -> float:
    """Evaluates MMLU subset accuracy."""
    from datasets import load_dataset
    print(f"  [Task] Evaluating MMLU ({max_samples} samples)...")
    try:
        dataset = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        # Fallback to high_school_biology if full MMLU takes too long to load
        dataset = load_dataset("cais/mmlu", "high_school_biology", split="test")
        
    correct = 0
    total = min(len(dataset), max_samples)
    for idx in range(total):
        item = dataset[idx]
        q = f"Question: {item['question']}\nAnswer:"
        choices = item["choices"]
        gold_idx = int(item["answer"])
        
        scores = [get_log_likelihood(model, tokenizer, q + " ", str(c), device) for c in choices]
        pred = int(np.argmax(scores))
        if pred == gold_idx:
            correct += 1
            
    acc = correct / total * 100.0
    return round(acc, 2)


def eval_inference_speed(
    model,
    tokenizer,
    device: torch.device,
    prompt_len: int = 64,
    gen_len: int = 128,
    warmup_iters: int = 3,
    eval_iters: int = 10
) -> Dict[str, float]:
    """
    Measures autoregressive generation speed:
    - Latency (ms per token generated)
    - Throughput (tokens generated per second)
    """
    print(f"  [Speed] Benchmarking inference throughput & latency ({gen_len} tokens)...")
    dummy_input = torch.randint(100, 20000, (1, prompt_len), device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            curr_input = dummy_input.clone()
            for _ in range(16):
                out = model(curr_input)
                next_token = torch.argmax(out.logits[:, -1:, :], dim=-1)
                curr_input = torch.cat([curr_input, next_token], dim=-1)
                
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(eval_iters):
            curr_input = dummy_input.clone()
            t0 = time.perf_counter()
            for _ in range(gen_len):
                out = model(curr_input)
                next_token = torch.argmax(out.logits[:, -1:, :], dim=-1)
                curr_input = torch.cat([curr_input, next_token], dim=-1)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    avg_time = float(np.mean(times))
    tokens_per_sec = gen_len / avg_time
    ms_per_token = (avg_time / gen_len) * 1000.0
    
    return {
        "throughput_tokens_per_sec": round(tokens_per_sec, 2),
        "latency_ms_per_token": round(ms_per_token, 2)
    }


# ==============================================================================
# 4. Main Evaluation Orchestrator & Visualization
# ==============================================================================

def plot_benchmark_results(results: Dict[str, Dict[str, Any]], output_path: str):
    """Draws grouped bar chart comparing all evaluated models across paper benchmarks."""
    models = list(results.keys())
    if not models:
        return
    
    # Complete 10 Core Downstream Tasks from Paper Section 4.1 + Average
    task_keys = [
        "hellaswag", "arc_challenge", "sciq", "openbookqa", "boolq",
        "winogrande", "cqa", "copa", "siqa", "mmlu", "avg_accuracy"
    ]
    task_labels = [
        "HellaSwag", "ARC-Chal", "SciQ", "OpenBookQA", "BoolQ",
        "WinoGrande", "C-QA", "COPA", "Social-IQa", "MMLU", "AVG-10"
    ]
    
    available_tasks = []
    available_labels = []
    for k, label in zip(task_keys, task_labels):
        if any(k in results[m] for m in models):
            available_tasks.append(k)
            available_labels.append(label)

    if not available_tasks:
        return

    n_tasks = len(available_tasks)
    x = np.arange(n_tasks)
    bar_width = 0.8 / len(models)
    
    colors = {
        "erc": "#d95f02",
        "vanilla": "#2b5c8f",
        "aoe": "#7570b3",
        "deepseek": "#1b9e77"
    }

    fig, ax = plt.subplots(figsize=(max(12, n_tasks * 1.3), 6))
    for i, m in enumerate(models):
        vals = [results[m].get(tk, 0.0) for tk in available_tasks]
        offset = (i - len(models)/2 + 0.5) * bar_width
        color = colors.get(m, None)
        ax.bar(x + offset, vals, width=bar_width, label=m.upper(), color=color, alpha=0.9, edgecolor="black", linewidth=0.6)

    ax.set_title("Downstream Benchmark Comparison (ICLR 2026 Paper Evaluation Tasks)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(available_labels, fontsize=10, rotation=15)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"--> Saved benchmark comparison plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MoE models across all datasets from the ICLR 2026 paper.")
    parser.add_argument("--checkpoints_dir", type=str, default="/kaggle/working",
                        help="Base directory containing checkpoints (e.g. /kaggle/working or /kaggle/input)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/eval_results",
                        help="Directory to save evaluation results and plots")
    parser.add_argument("--models", type=str, default="erc,vanilla,aoe,deepseek",
                        help="Comma-separated list of models to evaluate: erc,vanilla,aoe,deepseek")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated tasks to evaluate, or 'all' for full paper suite")
    parser.add_argument("--max_samples", type=int, default=300,
                        help="Max samples per benchmark task (-1 for full split)")
    parser.add_argument("--eval_speed", action="store_true", default=True,
                        help="Whether to benchmark inference speed and latency")
    parser.add_argument("--include_gsm8k", action="store_true", default=False,
                        help="Include GSM8K math reasoning (from Table 1)")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                        help="Log evaluation results, tables, and comparison charts to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="moe-erc-iclr2026-eval",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="moe_benchmarks_eval",
                        help="WandB run name")
    args = parser.parse_args()

    # Optional WandB initialization
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
            print(f"--> WandB initialized: {args.wandb_project} / {args.wandb_run_name}")
        except Exception as e:
            print(f"[Warning] Failed to initialize WandB: {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running MoE Evaluation on Device: {device} ===")

    print("Loading tokenizer allenai/OLMoE-1B-7B-0924...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    models_to_eval = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    overall_results: Dict[str, Dict[str, Any]] = {}

    # Define all 10 downstream paper tasks
    all_downstream_tasks = [
        ("hellaswag", eval_hellaswag),
        ("arc_challenge", eval_arc_challenge),
        ("sciq", eval_sciq),
        ("openbookqa", eval_openbookqa),
        ("boolq", eval_boolq),
        ("winogrande", eval_winogrande),
        ("cqa", eval_commonsenseqa),
        ("copa", eval_copa),
        ("siqa", eval_social_i_qa),
        ("mmlu", eval_mmlu),
    ]
    if args.include_gsm8k:
        all_downstream_tasks.append(("gsm8k", eval_gsm8k))

    selected_task_names = [t.strip().lower() for t in args.tasks.split(",")] if args.tasks != "all" else None
    eval_tasks = [
        (name, fn) for (name, fn) in all_downstream_tasks 
        if selected_task_names is None or name in selected_task_names
    ]

    for m_type in models_to_eval:
        ckpt_path = find_checkpoint_path(args.checkpoints_dir, m_type)
        if ckpt_path is None:
            print(f"[Skip] Checkpoint not found for model: {m_type.upper()} in {args.checkpoints_dir}")
            continue

        print(f"\n=======================================================")
        print(f" Evaluating Model: {m_type.upper()}")
        print(f" Checkpoint: {ckpt_path}")
        print(f"=======================================================")

        try:
            model = load_moe_model(m_type, args.checkpoints_dir, tokenizer, device)
        except Exception as e:
            print(f"[Error] Failed to load model {m_type}: {e}")
            continue

        res: Dict[str, Any] = {}

        # 1. Wikitext-2 PPL
        if selected_task_names is None or "wikitext" in selected_task_names:
            try:
                wt_res = eval_wikitext2(model, tokenizer, device, max_tokens=30000)
                res.update(wt_res)
                print(f"    --> Wikitext-2 PPL: {wt_res['wikitext_ppl']} (Loss: {wt_res['wikitext_loss']})")
            except Exception as e:
                print(f"  [Warning] Wikitext evaluation failed: {e}")

        # 2. Downstream Tasks
        task_scores = []
        for task_name, eval_fn in eval_tasks:
            try:
                acc = eval_fn(model, tokenizer, device, max_samples=args.max_samples)
                res[task_name] = acc
                task_scores.append(acc)
                print(f"    --> {task_name.upper()}: {acc:.2f}%")
            except Exception as e:
                print(f"  [Warning] Task {task_name} evaluation skipped/failed: {e}")

        if task_scores:
            avg_acc = round(float(np.mean(task_scores)), 2)
            res["avg_accuracy"] = avg_acc
            print(f"    ==> AVERAGE DOWNSTREAM ACCURACY: {avg_acc:.2f}%")

        # 3. Inference Speed
        if args.eval_speed:
            try:
                speed_res = eval_inference_speed(model, tokenizer, device)
                res.update(speed_res)
                print(f"    --> Speed: {speed_res['throughput_tokens_per_sec']} tokens/s ({speed_res['latency_ms_per_token']} ms/token)")
            except Exception as e:
                print(f"  [Warning] Speed benchmark failed: {e}")

        overall_results[m_type] = res
        
        # Log individual model metrics to WandB
        if wandb_run is not None:
            import wandb
            m_metrics = {f"{m_type}/{k}": v for k, v in res.items() if isinstance(v, (int, float))}
            wandb.log(m_metrics)

        # Free GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save JSON summary
    summary_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(summary_file, "w") as f:
        json.dump(overall_results, f, indent=2)
    print(f"\n--> All results saved to: {summary_file}")

    # Generate Markdown Table
    md_file = os.path.join(args.output_dir, "evaluation_summary.md")
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# MoE Architecture Benchmark Evaluation Results (ICLR 2026 Paper Suite)\n\n")
        f.write("| Model | Wikitext PPL (↓) | AVG Acc (↑) | HellaSwag | ARC-Chal | SciQ | OpenBook | BoolQ | WinoGrande | C-QA | COPA | SocialIQa | MMLU | Throughput (tok/s) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for m, d in overall_results.items():
            ppl = d.get("wikitext_ppl", "-")
            avg = d.get("avg_accuracy", "-")
            hs = d.get("hellaswag", "-")
            arc = d.get("arc_challenge", "-")
            sciq = d.get("sciq", "-")
            obqa = d.get("openbookqa", "-")
            boolq = d.get("boolq", "-")
            wg = d.get("winogrande", "-")
            cqa = d.get("cqa", "-")
            copa = d.get("copa", "-")
            siqa = d.get("siqa", "-")
            mmlu = d.get("mmlu", "-")
            tps = d.get("throughput_tokens_per_sec", "-")
            f.write(f"| **{m.upper()}** | {ppl} | **{avg}%** | {hs}% | {arc}% | {sciq}% | {obqa}% | {boolq}% | {wg}% | {cqa}% | {copa}% | {siqa}% | {mmlu}% | {tps} |\n")

    print(f"--> Summary table saved to: {md_file}")

    # Plot figure
    plot_file = os.path.join(args.output_dir, "benchmark_comparison.png")
    plot_benchmark_results(overall_results, plot_file)

    # Log summary table and figure to WandB
    if wandb_run is not None:
        import wandb
        # 1. WandB Table
        columns = ["Model", "Wikitext PPL", "AVG Acc (%)", "HellaSwag", "ARC-Chal", "SciQ", "OpenBook", "BoolQ", "WinoGrande", "C-QA", "COPA", "SocialIQa", "MMLU", "Throughput (tok/s)"]
        wb_table = wandb.Table(columns=columns)
        for m, d in overall_results.items():
            row = [
                m.upper(),
                d.get("wikitext_ppl", None),
                d.get("avg_accuracy", None),
                d.get("hellaswag", None),
                d.get("arc_challenge", None),
                d.get("sciq", None),
                d.get("openbookqa", None),
                d.get("boolq", None),
                d.get("winogrande", None),
                d.get("cqa", None),
                d.get("copa", None),
                d.get("siqa", None),
                d.get("mmlu", None),
                d.get("throughput_tokens_per_sec", None)
            ]
            wb_table.add_data(*row)
        wandb.log({"evaluation_summary_table": wb_table})

        # 2. WandB Image
        if os.path.exists(plot_file):
            wandb.log({"benchmark_comparison_plot": wandb.Image(plot_file)})

        wandb.finish()
        print("--> Logged all tables, charts, and metrics to WandB successfully!")


if __name__ == "__main__":
    main()
