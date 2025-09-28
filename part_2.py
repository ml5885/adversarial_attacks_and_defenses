import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import torch
import csv
import random
import numpy as np

import train
import model

plt.rcParams.update({'font.family': 'serif'})

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_experiment(max_steps, log_every, num_trials=10, base_seed=0):
    """Train baseline, FGSM and TRADES models and evaluate them across multiple trials.

    Args:
        max_steps: Maximum number of optimisation steps for each training routine.
        log_every: Print training progress every this many steps.
        num_trials: Number of independent trials to run.
        base_seed: Base random seed; each trial uses base_seed + trial_idx.

    Returns:
        A nested dictionary mapping each training regime ("baseline", "fgsm", "trades")
        to a mapping from epsilon to a list of accuracies across trials.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare data
    train_loader, test_loader = train.get_data_loaders(batch_size=50, download=True)
    # Evaluate on a fixed set of epsilon values
    eval_eps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = {}

    for trial_idx in range(num_trials):
        # Ensure different randomness per trial
        seed = (base_seed or 0) + trial_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"\n=== Trial {trial_idx + 1}/{num_trials} (seed={seed}) ===", flush=True)

        # Baseline (standard training)
        print("Training baseline model...", flush=True)
        baseline_model = model.ConvNet()
        baseline_model = train.train_baseline(
            baseline_model,
            train_loader,
            device=device,
            lr=1e-4,
            max_steps=max_steps,
            log_every=log_every,
        )
        print("Evaluating baseline model...", flush=True)
        accs = train.evaluate_fgsm(
            baseline_model, test_loader, device=device, epsilons=eval_eps
        )
        for eps, acc in accs.items():
            results.setdefault("baseline", {}).setdefault(eps, []).append(acc)

        # FGSM adversarial training
        print("Training FGSM adversarially trained model...", flush=True)
        fgsm_model = model.ConvNet()
        fgsm_model = train.train_fgsm(
            fgsm_model,
            train_loader,
            device=device,
            epsilon_train=0.3,
            lr=1e-4,
            max_steps=max_steps,
            log_every=log_every,
        )
        print("Evaluating FGSM adversarially trained model...", flush=True)
        accs = train.evaluate_fgsm(
            fgsm_model, test_loader, device=device, epsilons=eval_eps
        )
        for eps, acc in accs.items():
            results.setdefault("fgsm", {}).setdefault(eps, []).append(acc)

        # TRADES training
        print("Training TRADES adversarially trained model...", flush=True)
        trades_model = model.ConvNet()
        trades_model = train.train_trades(
            trades_model,
            train_loader,
            device=device,
            epsilon_train=0.3,
            beta=6.0,
            lr=1e-4,
            max_steps=max_steps,
            log_every=log_every,
        )
        print("Evaluating TRADES model...", flush=True)
        accs = train.evaluate_fgsm(
            trades_model, test_loader, device=device, epsilons=eval_eps
        )
        for eps, acc in accs.items():
            results.setdefault("trades", {}).setdefault(eps, []).append(acc)

    return results


def plot_results(results, out_path, confidence=0.99):
    curve_styles = {
        "baseline": {"label": "Baseline", "color": "#0072B2", "marker": "o"},
        "fgsm": {"label": "FGSM-trained", "color": "#009E73", "marker": "s"},
        "trades": {"label": "TRADES", "color": "#D55E00", "marker": "^"},
    }
    # Map common confidence levels to z-scores (normal approx)
    z_map = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}
    z = z_map.get(confidence, 2.576)

    fig, ax = plt.subplots(figsize=(8, 6))
    for key, acc_dict in results.items():
        eps_values = sorted(acc_dict.keys())
        means, lowers, uppers = [], [], []
        for eps in eps_values:
            vals = np.asarray(acc_dict[eps] if isinstance(acc_dict[eps], (list, tuple)) else [acc_dict[eps]], dtype=float)
            n = max(len(vals), 1)
            m = float(vals.mean())
            s = float(vals.std(ddof=1)) if n > 1 else 0.0
            ci = z * (s / np.sqrt(n)) if n > 1 else 0.0
            means.append(m)
            lowers.append(max(0.0, m - ci))
            uppers.append(min(1.0, m + ci))
        style = curve_styles.get(key, {})
        ax.plot(
            eps_values,
            means,
            label=style.get("label", key),
            color=style.get("color", None),
            marker=style.get("marker", None),
            linestyle="-",
        )
        ax.fill_between(
            eps_values, lowers, uppers, color=style.get("color", None), alpha=0.15, linewidth=0
        )
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlabel("Epsilon", fontsize=24)
    ax.set_ylabel("Accuracy", fontsize=24)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, max(max(k for acc_dict in results.values() for k in acc_dict.keys()), 0.3))
    ax.grid(True, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.legend(loc="best", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def save_results_to_csv(results: Dict[str, Dict[float, list]], csv_path: str) -> None:
    """Save per-trial results to a CSV with columns: regime, epsilon, accuracy, trial."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["regime", "epsilon", "accuracy", "trial"]) 
        for regime, acc_dict in results.items():
            for eps, acc_list in acc_dict.items():
                for trial_idx, acc in enumerate(acc_list):
                    writer.writerow([regime, float(eps), float(acc), int(trial_idx)])
    print(f"Saved results to {csv_path}")


def load_results_from_csv(csv_path: str) -> Dict[str, Dict[float, list]]:
    """Load per-trial results and aggregate them as lists for each epsilon.

    Supports both the new format with a 'trial' column and the legacy format
    without it (in which case there is one value per epsilon).
    """
    results: Dict[str, Dict[float, list]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        has_trial = "trial" in reader.fieldnames if reader.fieldnames else False
        for row in reader:
            regime = row["regime"]
            eps = float(row["epsilon"]) if row.get("epsilon") is not None else 0.0
            acc = float(row["accuracy"]) if row.get("accuracy") is not None else 0.0
            results.setdefault(regime, {}).setdefault(eps, [])
            # Append once per row; legacy rows will yield a single-entry list
            results[regime][eps].append(acc)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial training experiments on MNIST.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100_000,
        help="Maximum number of optimisation steps for each training regime (default: 100000)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=500,
        help="Print training progress every N steps (default: 500)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of independent trials to run and aggregate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (each trial uses seed+trial_idx)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Confidence level for CI shading in the plot (default: 0.99)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "part_2"),
        help="Directory in which to save results and plots.",
    )
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Load results from CSV and generate plot without re-running training.",
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        default=None,
        help="Path to results CSV (default: <output-dir>/results.csv)",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    if args.results_csv is None:
        args.results_csv = os.path.join(args.output_dir, "results.csv")

    if args.analysis:
        if not os.path.exists(args.results_csv):
            print(f"Results CSV not found at {args.results_csv}. Run without --analysis to generate it.")
            return
        results = load_results_from_csv(args.results_csv)
        plot_path = os.path.join(args.output_dir, "accuracy_vs_epsilon.png")
        plot_results(results, plot_path, confidence=args.confidence)
        return

    results = run_experiment(max_steps=args.max_steps, log_every=args.log_every, num_trials=args.trials, base_seed=args.seed)

    # Print aggregated results table (mean +/- 99% CI)
    print("\nAggregated accuracy results:", flush=True)
    z = 2.576  # 99% normal z-score
    for name, acc_dict in results.items():
        print(f"  {name}:", flush=True)
        for eps in sorted(acc_dict.keys()):
            vals = np.asarray(acc_dict[eps], dtype=float)
            n = len(vals)
            m = float(vals.mean())
            s = float(vals.std(ddof=1)) if n > 1 else 0.0
            ci = z * (s / np.sqrt(n)) if n > 1 else 0.0
            print(f"    epsilon={eps:.1f}: mean={m:.4f}, ci99=±{ci:.4f}", flush=True)

    # Save results and plot
    save_results_to_csv(results, args.results_csv)
    plot_path = os.path.join(args.output_dir, "accuracy_vs_epsilon.png")
    plot_results(results, plot_path, confidence=args.confidence)


if __name__ == "__main__":
    main()
