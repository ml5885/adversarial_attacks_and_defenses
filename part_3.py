import argparse
import csv
import json
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gcg.nanogcg as nanogcg
from gcg.nanogcg import GCGConfig

plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
OUTPUT_DIR = "results/part_3/"
BEHAVIORS_FILE = "behaviors.json"

MODEL_DISPLAY_NAME = {
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Instruct v0.2",
    "meta-llama/Llama-3.2-1B-Instruct": "LLaMA 3.2 1B Instruct",
}

def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def is_successful(response, target):
    return response.strip().lower().startswith(target.strip().lower())

def plot_losses(steps, losses, success_step=None, title="Loss Curve", xlabel="Step", ylabel="Loss", save_path="loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, label='Loss')
    if success_step is not None and success_step in steps:
        idx = steps.index(success_step)
        plt.scatter([steps[idx]], [losses[idx]], marker='*', s=240, edgecolors='k', zorder=3,
                    label=f'First non-refusal @ {success_step}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_asr_vs_iterations(asr_data, model_names, title, save_path):
    plt.figure(figsize=(10, 6))
    iterations = sorted(list(asr_data.keys()))
    steps = [(i + 1) * 20 for i in iterations]

    for i, model_name in enumerate(model_names):
        # Shorten model name for legend
        label_name = model_name.split('/')[-1]
        asr_values = [asr_data[it][i] for it in iterations]
        plt.plot(steps, asr_values, marker='o', linestyle='-', label=label_name)

    plt.title(title)
    plt.xlabel("Iteration Step")
    plt.ylabel("Attack Success Rate (ASR)")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_asr_bar_chart(final_asr, model_names, title, save_path):
    plt.figure(figsize=(8, 6))
    
    seen_models = model_names[:-1]
    unseen_model = model_names[-1]
    
    seen_asr = np.mean([final_asr[model] for model in seen_models])
    unseen_asr = final_asr[unseen_model]

    labels = [f"Seen Models (Avg)\n{', '.join([m.split('/')[-1] for m in seen_models])}", f"Unseen Model\n{unseen_model.split('/')[-1]}"]
    asr_values = [seen_asr, unseen_asr]
    
    bars = plt.bar(labels, asr_values, color=['skyblue', 'salmon'])
    plt.ylabel('Attack Success Rate (ASR)')
    plt.title(title)
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    plt.savefig(save_path)
    plt.close()

def generate(model, tokenizer, device, text, max_new_tokens=100):
    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_token_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

def run_task_1(args):
    if args.analysis:
        # load results from csv for plotting
        output_dir = os.path.join(args.output_dir, "task1")
        csv_path = os.path.join(output_dir, "task1_loss_log.csv")
        if not os.path.exists(csv_path):
            print(f"Error: CSV log file not found at '{csv_path}' for analysis.")
            return
        logged_steps, logged_losses, logged_success = [], [], []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                logged_steps.append(int(row["iteration"]))
                logged_losses.append(float(row["loss"]))
                logged_success.append(int(row["first_non_refusal_here"]))
        first_success_step = None
        for step, success in zip(logged_steps, logged_success):
            if success == 1:
                first_success_step = step
                break
        print(f"Loaded logged data from '{csv_path}' for analysis.")
    else:
        print("--- Running Task 1: Single-Model Attack ---")
        with open(BEHAVIORS_FILE, "r") as f:
            behaviors = json.load(f)
        
        behavior_obj = behaviors[0]
        message = behavior_obj["behavior"]
        target = behavior_obj["target_prefix"]
        print(f"Attacking with behavior: '{message}'")

        output_dir = os.path.join(args.output_dir, "task1")
        ensure_dir(output_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(args.model_ids[0], torch_dtype=dtype).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_ids[0])
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        default_response = generate(model, tokenizer, device, message)
        print(f"--- Default Model Output (no suffix) ---\n{default_response}")

        config = GCGConfig(
            num_steps=args.num_steps,
            search_width=args.search_width,
            topk=args.topk,
            seed=args.seed,
            verbosity=args.verbosity,
            optim_str_init="x x x x x x x x x x x x x x x x x x x x",
        )
        
        result = nanogcg.run(model, tokenizer, message, target, config)

        first_success_step = None
        logged_steps, logged_losses, logged_success = [], [], []

        for i, suffix in enumerate(result.strings):
            iteration_step = i + 1
            if iteration_step == 1 or iteration_step % 20 == 0:
                logged_steps.append(iteration_step)
                logged_losses.append(result.losses[i])

                success = False
                if first_success_step is None:
                    response = generate(model, tokenizer, device, f"{message}{suffix}")
                    success = is_successful(response, target)
                    if success:
                        first_success_step = iteration_step
                        print(f"Success first detected at step: {first_success_step}")
                logged_success.append(1 if success and iteration_step == first_success_step else 0)

        csv_path = os.path.join(output_dir, "task1_loss_log.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "loss", "first_non_refusal_here"])
            for s, l, ok in zip(logged_steps, logged_losses, logged_success):
                writer.writerow([s, l, ok])
        print(f"Saved checkpoint log to {csv_path}")

        final_suffix = result.best_string
        final_response = generate(model, tokenizer, device, f"{message}{final_suffix}")

        print(f"Model: {args.model_ids[0]}")
        print(f"Behavior: {message}")
        print(f"Target: {target}")
        print(f"--- Final Suffix ---\n{final_suffix}")
        print(f"--- Final Model Output ---\n{final_response}")

        with open(os.path.join(output_dir, "task1_results.txt"), "w") as f:
            f.write(f"Model: {args.model_ids[0]}\n")
            f.write(f"Behavior: {message}\n")
            f.write(f"Target: {target}\n\n")
            f.write(f"--- Default Model Output (no suffix) ---\n{default_response}\n\n")
            f.write(f"--- Final Suffix ---\n{final_suffix}\n\n")
            f.write(f"--- Final Model Output ---\n{final_response}\n")
        print("Saved final suffix and model output.")

    plot_path = os.path.join(output_dir, "task1_loss_curve.png")
    plot_losses(
        logged_steps,
        logged_losses,
        success_step=first_success_step,
        save_path=plot_path,
        title=f"GCG Loss vs. Iterations ({MODEL_DISPLAY_NAME.get(args.model_ids[0], args.model_ids[0])})"
    )
    print(f"Saved loss plot to {plot_path}")

def run_task_2b(args):
    if args.analysis:
        # Load results from csv for plotting
        output_dir = os.path.join(args.output_dir, "task2b")
        csv_path = os.path.join(output_dir, "task2b_asr_over_time.csv")
        if not os.path.exists(csv_path):
            print(f"Error: CSV log file not found at '{csv_path}' for analysis.")
            return
        asr_counts = {}
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            model_names = reader.fieldnames[1:]
            for row in reader:
                iteration = int(row["iteration"])
                asr_counts[iteration] = [float(row[model]) for model in model_names]
        print(f"Loaded ASR counts from '{csv_path}' for analysis.")
    else:
        print("--- Running Task 2B: Transferability via Model Ensembling ---")

        output_dir = os.path.join(args.output_dir, "task2b")
        ensure_dir(output_dir)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        if len(args.model_ids) != 2:
            raise ValueError("Task 2B requires exactly two --model-ids (same tokenizer family).")

        model_id_a, model_id_b = args.model_ids
        transfer_id = args.transfer_model_id

        # Load ensemble models (shared tokenizer)
        model_a = AutoModelForCausalLM.from_pretrained(model_id_a, torch_dtype=dtype).to(device).eval()
        model_b = AutoModelForCausalLM.from_pretrained(model_id_b, torch_dtype=dtype).to(device).eval()
        tok_shared = AutoTokenizer.from_pretrained(model_id_a)
        if tok_shared.pad_token_id is None: tok_shared.pad_token_id = tok_shared.eos_token_id

        # Assert same tokenizer vocab
        tok_b = AutoTokenizer.from_pretrained(model_id_b)
        if tok_b.vocab_size != tok_shared.vocab_size:
            raise ValueError("The two ensemble models must share the same tokenizer/vocab. Choose models from the same family.")
        # We will optimize under tok_shared (assumed compatible with model_b family)

        # Held-out (transfer) model + its tokenizer (can differ)
        transfer_model = AutoModelForCausalLM.from_pretrained(transfer_id, torch_dtype=dtype).to(device).eval()
        transfer_tok = AutoTokenizer.from_pretrained(transfer_id)
        if transfer_tok.pad_token_id is None: transfer_tok.pad_token_id = transfer_tok.eos_token_id

        # Load behaviors and pick first 10 (or fewer if file has less)
        with open(BEHAVIORS_FILE, "r") as f:
            behaviors = json.load(f)
        num_behaviors = min(10, len(behaviors))
        if num_behaviors == 0:
            raise ValueError("behaviors.json is empty; need at least 1 behavior.")

        # For ASR over time (every 20 iters)
        asr_counts = {}  # iter -> [seen_model_a_count, seen_model_b_count, unseen_transfer_count]
        steps_logged = set()

        # Joint optimization config (log every 20 via our evaluation loop)
        config = GCGConfig(
            num_steps=args.num_steps,
            search_width=args.search_width,
            topk=args.topk,
            seed=args.seed,
            verbosity=args.verbosity,
            optim_str_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",  # 20 tokens
        )

        for idx in range(num_behaviors):
            beh = behaviors[idx]
            message = beh["behavior"]
            target  = beh["target_prefix"]

            # Optional: verify baseline refusals
            base_a = generate(model_a, tok_shared, device, message)
            base_b = generate(model_b, tok_shared, device, message)
            base_t = generate(transfer_model, transfer_tok, device, message)

            # Run joint optimization on (model_a, model_b)
            result = nanogcg.run_ensemble([model_a, model_b], tok_shared, message, target, config)

            # Evaluate every 20 iterations across all three models
            for i, suffix in enumerate(result.strings):
                step = i + 1
                if step % 20 != 0:
                    continue

                # Prepare counters
                if step not in asr_counts:
                    asr_counts[step] = [0, 0, 0]
                steps_logged.add(step)

                # Evaluate success on both seen models (use shared tokenizer that training used)
                resp_a = generate(model_a, tok_shared, device, f"{message}{suffix}")
                resp_b = generate(model_b, tok_shared, device, f"{message}{suffix}")
                # Evaluate on held-out transfer model with its own tokenizer
                resp_t = generate(transfer_model, transfer_tok, device, f"{message}{suffix}")

                asr_counts[step][0] += 1 if is_successful(resp_a, target) else 0
                asr_counts[step][1] += 1 if is_successful(resp_b, target) else 0
                asr_counts[step][2] += 1 if is_successful(resp_t, target) else 0

    # Convert counts to rates
    asr_data = {}
    for step in sorted(list(steps_logged)):
        seen_a = asr_counts[step][0] / num_behaviors
        seen_b = asr_counts[step][1] / num_behaviors
        unseen = asr_counts[step][2] / num_behaviors
        asr_data[step//20 - 1] = [seen_a, seen_b, unseen]  # index by checkpoint (0,1,2,...) for plotting helper

    # Save CSV of ASR over time
    csv_path = os.path.join(output_dir, "task2b_asr_over_time.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", model_id_a, model_id_b, transfer_id])
        for step in sorted(asr_counts.keys()):
            row = [step, asr_counts[step][0] / num_behaviors, asr_counts[step][1] / num_behaviors, asr_counts[step][2] / num_behaviors]
            writer.writerow(row)
    print(f"Saved Task 2B ASR log to {csv_path}")

    # Plots
    model_names = [model_id_a, model_id_b, transfer_id]

    line_path = os.path.join(output_dir, "task2b_asr_vs_iterations.png")
    plot_asr_vs_iterations(asr_data, model_names, title="Task 2B: ASR vs Iterations (Seen vs Unseen)", save_path=line_path)
    print(f"Saved Task 2B ASR vs Iterations plot to {line_path}")

    # Final bar chart: use the last checkpoint
    if len(asr_data) > 0:
        last_idx = max(asr_data.keys())
        final_vals = asr_data[last_idx]
        final_asr = {model_names[i]: final_vals[i] for i in range(3)}
        bar_path = os.path.join(output_dir, "task2b_asr_seen_vs_unseen.png")
        plot_asr_bar_chart(final_asr, model_names, title="Task 2B: Seen (Avg) vs Unseen ASR", save_path=bar_path)
        print(f"Saved Task 2B Seen vs Unseen bar chart to {bar_path}")

def main():
    parser = argparse.ArgumentParser(description="Part 3: Attacking Aligned LMs")
    parser.add_argument("--task", type=str, required=True, choices=['task1', 'task2b'], help="Task to run.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory for outputs.")
    
    parser.add_argument("--model-ids", nargs='+', default=["mistralai/Mistral-7B-Instruct-v0.2"], help="For task1, one model. For task2b, two ensemble models.")
    parser.add_argument("--transfer-model-id", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Held-out model for task2b transfer.")

    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--search-width", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbosity", type=str, default="WARNING")
    
    parser.add_argument("--analysis", action='store_true', help="If set, only run analysis on existing logs/plots.")

    args = parser.parse_args()

    if not os.path.exists(BEHAVIORS_FILE):
        print(f"Error: Behaviors file not found at '{BEHAVIORS_FILE}'")
        return

    if args.task == 'task1':
        if len(args.model_ids) != 1:
            parser.error("Task 1 requires exactly one model specified with --model-ids.")
        run_task_1(args)
    elif args.task == 'task2b':
        if len(args.model_ids) < 2:
            parser.error("Task 2B requires exactly two models specified with --model-ids.")
        run_task_2b(args)

if __name__ == "__main__":
    main()