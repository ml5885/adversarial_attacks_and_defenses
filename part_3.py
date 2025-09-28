import argparse
import csv
import os
import random
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import requests

from tabulate import tabulate

import gcg

plt.rcParams.update({'font.family': 'serif'})


HARM_BENCH_VAL_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_val.csv"
HARM_BENCH_TEST_URL = "https://raw.githubusercontent.com/centerforaisafety/HarmBench/refs/heads/main/data/behavior_datasets/harmbench_behaviors_text_test.csv"


def download_behaviors(url: str) -> List[str]:
    """Download a list of harmful behaviour prompts from a CSV file.

    The HarmBench datasets contain a header row with a column called
    behavior. Each subsequent row contains one harmful query.

    Args:
        url: HTTP URL pointing to the CSV file.

    Returns:
        A list of behaviour strings.
    """
    resp = requests.get(url)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    reader = csv.DictReader(lines)
    behaviors = []
    for row in reader:
        behaviour = row.get("Behavior")
        if behaviour:
            behaviors.append(behaviour.strip())
    return behaviors


def plot_loss_trace(loss_trace, title, out_path):
    """Plot the optimisation loss over iterations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(loss_trace)), loss_trace, color="#0072B2", linestyle="-", marker="")
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Loss (negative log prob)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, linewidth=0.5, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved loss trace plot to {out_path}")


def plot_bar_asr(asr, title, out_path):
    """Create a bar chart of attack success rates per model."""
    fig, ax = plt.subplots(figsize=(8, 6))
    models = list(asr.keys())
    rates = [asr[m] for m in models]
    ax.bar(models, rates, color=["#0072B2" if m != models[-1] else "#D55E00" for m in models])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack Success Rate", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(axis='y', linewidth=0.5, linestyle="--", alpha=0.7)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved ASR bar plot to {out_path}")


def plot_average_loss_traces(traces, title, out_path):
    """Plot the average loss trace per model across behaviors.
    
    Traces may have different lengths; averaging at each iteration
    only uses available traces.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
    for i, (model_name, model_traces) in enumerate(traces.items()):
        # Find the maximum length among traces for this model
        max_len = max(len(t) for t in model_traces)
        mean_trace = []
        std_trace = []
        for step in range(max_len):
            vals = [t[step] for t in model_traces if len(t) > step]
            if vals:
                mean_trace.append(np.mean(vals))
                std_trace.append(np.std(vals))
            else:
                # No more traces at this step
                break
        x = list(range(len(mean_trace)))
        color = colors[i % len(colors)]
        ax.plot(x, mean_trace, label=model_name, color=color)
        ax.fill_between(x, np.array(mean_trace) - np.array(std_trace), np.array(mean_trace) + np.array(std_trace), color=color, alpha=0.2)
    ax.set_xlabel("Iteration", fontsize=14)
    ax.set_ylabel("Loss (negative log prob)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, linewidth=0.5, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved average loss trace plot to {out_path}")


def save_results_csv(results, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not results:
        return
    # Determine field names
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Saved CSV results to {csv_path}")


def load_results_csv(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def main():
    parser = argparse.ArgumentParser(description="Run GCG adversarial suffix experiments (Part 3).")
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Only generate plots from an existing CSV; do not run experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "part_3"),
        help="Directory to save results and plots."
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Path to a CSV file with saved results (for analysis)."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B",
        help="Comma-separated list of training models for transferability."
    )
    parser.add_argument(
        "--heldout",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Held-out model for evaluating transferability."
    )
    parser.add_argument(
        "--num-behaviors",
        type=int,
        default=10,
        help="Number of behaviors to sample for the transferability experiment."
    )
    parser.add_argument(
        "--per-model-steps",
        type=int,
        default=100,
        help="Number of GCG iterations per model in the transfer attack."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.results_csv is None:
        args.results_csv = os.path.join(args.output_dir, "results.csv")

    if args.analysis:
        # Analysis mode: load CSV and generate plots
        if not os.path.exists(args.results_csv):
            print(f"Results CSV not found at {args.results_csv}. Run without --analysis to generate it.")
            return
        results = load_results_csv(args.results_csv)
        # Compute ASR per model
        models = [key[len("success_"):] for key in results[0].keys() if key.startswith("success_")]
        asr = {}
        traces = {m: [] for m in models if m in args.models.split(",")}  # only training models
        for m in models:
            success_vals = [row[f"success_{m}"] == "True" for row in results]
            asr[m] = sum(success_vals) / len(success_vals) if success_vals else 0.0
        # Plot bar chart
        bar_title = "Attack Success Rate by Model (Transferability)"
        bar_out = os.path.join(args.output_dir, "asr_bar.png")
        plot_bar_asr(asr, bar_title, bar_out)
        # Plot average loss traces if present
        for row in results:
            for m in traces.keys():
                # Each loss trace is stored as a semi-colon separated string of floats
                loss_key = f"loss_trace_{m}"
                if loss_key in row and row[loss_key]:
                    trace = [float(x) for x in row[loss_key].split(";") if x]
                    traces[m].append(trace)
        if any(len(v) > 0 for v in traces.values()):
            loss_title = "Average GCG Loss vs. Iteration (Training Models)"
            loss_out = os.path.join(args.output_dir, "loss_traces.png")
            plot_average_loss_traces(traces, loss_title, loss_out)
        # Also print a summary table
        table_data = []
        for m in models:
            table_data.append([m, f"{asr[m]:.2f}"])
        print("\nAttack Success Rate (ASR) per model:")
        print(tabulate(table_data, headers=["Model", "ASR"], tablefmt="grid"))
        return

    # Experiment mode: run tasks and save results
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Download behaviour lists
    print("Downloading behaviour datasets...")
    val_behaviors = download_behaviors(HARM_BENCH_VAL_URL)
    test_behaviors = download_behaviors(HARM_BENCH_TEST_URL)

    if not val_behaviors or not test_behaviors:
        print("Error: Could not load behaviour datasets.")
        return
    
    # Sample behaviors
    num_behaviors = min(args.num_behaviors, len(val_behaviors), len(test_behaviors))
    random.shuffle(val_behaviors)
    random.shuffle(test_behaviors)
    selected_train = val_behaviors[:num_behaviors]
    selected_test = test_behaviors[:num_behaviors]

    training_models = [m.strip() for m in args.models.split(",") if m.strip()]
    heldout_model = args.heldout

    results_rows = []
    loss_traces_per_model: Dict[str, List[List[float]]] = {m: [] for m in training_models}

    # Task 1: Single prompt attack on the first training behaviour using the first training model
    single_message = selected_train[0]
    # We set a generic target prefix that encourages the model to answer affirmatively
    single_target = "Sure, here is the information you requested:\n\n"
    print(f"\n[Task 1] Running single-prompt attack on model {training_models[0]}...")
    res_single = gcg.run_gcg_single(
        message=single_message,
        target=single_target,
        model_name=training_models[0],
        num_steps=200,
        search_width=256,
        topk=128,
        n_replace=1,
        seed=args.seed,
    )
    suffix_single = res_single["optim_str"]
    loss_trace_single = res_single.get("loss_trace", [])
    # Generate final response on the attacking model
    model_single, tok_single = gcg.load_qwen_model(training_models[0])
    response_single = gcg.generate_model_response(model_single, tok_single, single_message, suffix_single)
    # Save loss trace plot
    loss_plot_path = os.path.join(args.output_dir, "task1_loss.png")
    plot_loss_trace(loss_trace_single, "GCG Optimisation Loss (Task 1)", loss_plot_path)
    # Save suffix and response to a text file
    with open(os.path.join(args.output_dir, "task1_output.txt"), "w") as f:
        f.write(f"Message: {single_message}\n")
        f.write(f"Suffix: {suffix_single}\n\n")
        f.write("Model response:\n")
        f.write(response_single + "\n")
    print("Saved Task 1 suffix and response.")

    # Task 2: Transferability via model ensembling
    print(f"\n[Task 2] Running transferability experiment on {num_behaviors} behaviors...")
    for i in range(num_behaviors):
        behaviour = selected_train[i]
        test_behaviour = selected_test[i]
        target_prefix = "Sure, here is the information you requested:\n\n"
        # Train suffix on training models and evaluate on held-out
        attack_result = gcg.run_transfer_attack(
            message=behaviour,
            target=target_prefix,
            training_models=training_models,
            heldout_model=heldout_model,
            per_model_steps=args.per_model_steps,
            search_width=256,
            topk=128,
            n_replace=1,
            seed=args.seed + i,
        )
        suffix = attack_result["suffix"]
        # Evaluate on test prompt to avoid training on the same prompt as evaluation
        success_results = {}
        response_results = {}
        # Evaluate on each model for the test prompt
        for m in training_models + [heldout_model]:
            model, tok = gcg.load_qwen_model(m)
            resp = gcg.generate_model_response(model, tok, test_behaviour, suffix)
            response_results[m] = resp
            success_results[m] = not gcg.is_refusal(resp)
        # Accumulate traces
        for m, trace in attack_result["traces"].items():
            loss_traces_per_model[m].append(trace)
        # Build result row
        row: Dict[str, str] = {"behavior": test_behaviour, "suffix": suffix}
        for m in training_models + [heldout_model]:
            row[f"response_{m}"] = response_results[m]
            row[f"success_{m}"] = str(success_results[m])
            # Store loss trace for training models only
            if m in attack_result["traces"]:
                # Serialise list of floats to a semicolon separated string
                row[f"loss_trace_{m}"] = ";".join([f"{x:.6f}" for x in attack_result["traces"][m]])
        results_rows.append(row)

    # Compute ASR for each model
    asr = {}
    for m in training_models + [heldout_model]:
        successes = [row[f"success_{m}"] == "True" for row in results_rows]
        asr[m] = sum(successes) / len(successes) if successes else 0.0

    # Print ASR table
    table_rows = []
    for m in training_models + [heldout_model]:
        table_rows.append([m, f"{asr[m]:.2f}"])
    print("\nTransferability Attack Success Rates:")
    print(tabulate(table_rows, headers=["Model", "ASR"], tablefmt="grid"))

    # Save CSV
    save_results_csv(results_rows, args.results_csv)
    # Generate plots
    bar_title = "Attack Success Rate by Model (Transferability)"
    bar_out = os.path.join(args.output_dir, "asr_bar.png")
    plot_bar_asr(asr, bar_title, bar_out)
    # Plot average loss traces across behaviors for training models
    if any(len(v) > 0 for v in loss_traces_per_model.values()):
        loss_title = "Average GCG Loss vs. Iteration (Training Models)"
        loss_out = os.path.join(args.output_dir, "loss_traces.png")
        plot_average_loss_traces(loss_traces_per_model, loss_title, loss_out)

    print(f"\nAll tasks complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()