import argparse
import csv
import os
import json
from datetime import datetime, timezone

import torch
import nanogcg
from nanogcg import GCGConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 14})
OUTPUT_DIR = "results/part_3/"


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def plot_losses(losses, title="Loss Curve", xlabel="Step", ylabel="Loss", save_path="loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()


def generate(model, tokenizer, device, text, max_new_tokens = 200):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    # Decode only the new tokens
    new_token_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_token_ids, skip_special_tokens=True).lstrip()


def _ensure_results_csv(csv_path):
    ensure_dir(os.path.dirname(csv_path))
    header = [
        "run_id", "timestamp", "model_id", "message", "target",
        "num_steps", "search_width", "topk", "seed", "step", "loss"
    ]
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def save_task1_results_to_csv(csv_path, meta, losses):
    _ensure_results_csv(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        for step, loss in enumerate(losses):
            writer.writerow([
                meta["run_id"], meta["timestamp"], meta["model_id"], meta["message"], meta["target"],
                meta["num_steps"], meta["search_width"], meta["topk"], meta["seed"], step, float(loss)
            ])


def load_task1_results_from_csv(csv_path, run_id=None):
    runs = {}
    if not os.path.exists(csv_path):
        return runs
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["run_id"]
            if run_id is not None and rid != run_id:
                continue
            runs.setdefault(rid, {
                "model_id": row.get("model_id", ""),
                "message": row.get("message", ""),
                "target": row.get("target", ""),
                "num_steps": int(row.get("num_steps", 0)),
                "losses": []
            })
            step = int(row["step"]) if row.get("step") not in (None, "") else None
            loss = float(row["loss"]) if row.get("loss") not in (None, "") else None
            if step is not None and loss is not None:
                runs[rid]["losses"].append((step, loss))
    # sort by step
    for rid in runs:
        runs[rid]["losses"].sort(key=lambda x: x[0])
    return runs


def run_single_gcg(
    model_id,
    message,
    target,
    num_steps=500,
    search_width=512,
    topk=256,
    seed=42,
    verbosity="WARNING",
    output_dir=OUTPUT_DIR,
    results_csv=None,
):
    ensure_dir(output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    config = GCGConfig(
        num_steps=num_steps,
        search_width=search_width,
        topk=topk,
        seed=seed,
        verbosity=verbosity,
    )

    run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{run_ts}_{seed}"

    result = nanogcg.run(model, tokenizer, message, target, config)
    adv_suffix = result.best_string

    # Use chat template for generation
    messages_run = [
        {"role": "user", "content": f"{message}{adv_suffix}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages_run,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_response = generate(model, tokenizer, device, prompt)
    losses = result.losses

    # Generate default response without suffix
    default_messages = [
        {"role": "user", "content": message},
    ]
    default_prompt = tokenizer.apply_chat_template(
        default_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    default_response = generate(model, tokenizer, device, default_prompt)

    # Save per-step losses to CSV for analysis
    csv_path = results_csv # The path is now fully constructed in main()
    meta = {
        "run_id": run_id,
        "timestamp": run_ts,
        "model_id": model_id,
        "message": message,
        "target": target,
        "num_steps": num_steps,
        "search_width": search_width,
        "topk": topk,
        "seed": seed,
    }
    save_task1_results_to_csv(csv_path, meta, losses)

    # Save loss plot and artifacts
    plot_path = os.path.join(output_dir, f"loss_curve_{run_id}.png")
    plot_losses(losses, title="nanoGCG Loss Curve", xlabel="Step", ylabel="Loss", save_path=plot_path)

    with open(os.path.join(output_dir, f"results_{run_id}.txt"), "w") as f:
        f.write("=== Adversarial suffix ===\n")
        f.write(adv_suffix + "\n\n")
        f.write("=== Attacked prompt (chat-formatted) ===\n")
        f.write(prompt + "\n\n")
        f.write("=== Model response ===\n")
        f.write(model_response + "\n\n")
        f.write("=== Default model response ===\n")
        f.write(default_response + "\n")

    return {
        "run_id": run_id,
        "adv_suffix": adv_suffix,
        "prompt": prompt,
        "model_response": model_response,
        "default_response": default_response,
        "losses": losses,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "target_prefix": target,
    }


def run_analysis_mode(csv_path, output_dir, run_id=None):
    ensure_dir(output_dir)
    runs = load_task1_results_from_csv(csv_path, run_id)
    if not runs:
        print(f"No results found in {csv_path}.")
        return

    # If multiple runs, plot each and also create an aggregate page
    for rid, info in runs.items():
        steps_losses = info["losses"]
        steps = [s for s, _ in steps_losses]
        losses = [l for _, l in steps_losses]
        out_path = os.path.join(output_dir, f"loss_curve_{rid}.png")
        plot_losses(losses, title=f"nanoGCG Loss Curve (run {rid})", xlabel="Step", ylabel="Loss", save_path=out_path)
        print(f"Saved plot: {out_path}")

    # Also produce a combined plot
    plt.figure(figsize=(8, 5))
    for rid, info in runs.items():
        steps_losses = info["losses"]
        steps = [s for s, _ in steps_losses]
        losses = [l for _, l in steps_losses]
        plt.plot(steps, losses, label=rid)
    plt.title("nanoGCG Loss Curves")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)
    plt.grid()
    combined_path = os.path.join(output_dir, "loss_curve_all_runs.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"Saved plot: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Part 3: GCG-style suffix optimization and analysis")
    parser.add_argument("--analysis", action="store_true", help="Load CSV and regenerate loss plots without running optimization")
    parser.add_argument("--results-csv", type=str, default="results.csv", help="Filename for the results CSV within the output directory.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Directory to save outputs and CSV")
    parser.add_argument("--run-id", type=str, default=None, help="Specific run_id to plot in analysis mode (default: all runs)")
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="HF model id")
    parser.add_argument("--behavior", type=int, default=0, help="Index of behavior to run from behaviors.json")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--search-width", type=int, default=512)
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbosity", type=str, default="WARNING")

    args = parser.parse_args()

    ensure_dir(args.output_dir)
    
    results_csv_path = os.path.join(args.output_dir, args.results_csv)

    if args.analysis:
        if not os.path.exists(results_csv_path):
            print(f"Results CSV not found at {results_csv_path}. Run without --analysis to generate it.")
            return
        run_analysis_mode(results_csv_path, args.output_dir, run_id=args.run_id)
        return

    # Load behaviors from JSON
    with open("behaviors.json", "r") as f:
        behaviors = json.load(f)

    behavior = behaviors[args.behavior]
    message = behavior["behavior"]
    target = behavior["target_prefix"]
    print(f"\nRunning GCG for behavior: {message}")
    out = run_single_gcg(
        model_id=args.model_id,
        message=message,
        target=target,
        num_steps=args.num_steps,
        search_width=args.search_width,
        topk=args.topk,
        seed=args.seed,
        verbosity=args.verbosity,
        output_dir=args.output_dir,
        results_csv=results_csv_path,
    )

    print("=== Adversarial suffix ===")
    print(out["adv_suffix"]) 
    print("=== Target prefix ===")
    print(out["target_prefix"])
    print("\n=== Attacked prompt (chat-formatted) ===")
    print(out["prompt"])
    print("\n=== Model response ===")
    print(out["model_response"]) 
    print("\n=== Default model response ===")
    print(out["default_response"]) 
    print(f"\nSaved CSV: {out['csv_path']}")
    print(f"Saved plot: {out['plot_path']}")

if __name__ == "__main__":
    main()