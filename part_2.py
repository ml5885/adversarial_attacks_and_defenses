import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import torch
import csv

import train
import model

plt.rcParams.update({'font.family': 'serif'})

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_experiment(max_steps, log_every):
    """Train baseline, FGSM and TRADES models and evaluate them.

    Args:
        max_steps: Maximum number of optimisation steps for each
            training routine.
        log_every: Print training progress every this many steps.

    Returns:
        A nested dictionary mapping each training regime ("baseline",
        "fgsm", "trades") to a mapping from epsilon to accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Prepare data
    train_loader, test_loader = train.get_data_loaders(batch_size=50, download=True)
    # Evaluate on a fixed set of epsilon values
    eval_eps = [0.0, 0.1, 0.2, 0.3]

    results: Dict[str, Dict[float, float]] = {}

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
    results["baseline"] = train.evaluate_fgsm(
        baseline_model, test_loader, device=device, epsilons=eval_eps
    )

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
    results["fgsm"] = train.evaluate_fgsm(
        fgsm_model, test_loader, device=device, epsilons=eval_eps
    )

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
    results["trades"] = train.evaluate_fgsm(
        trades_model, test_loader, device=device, epsilons=eval_eps
    )

    return results


def plot_results(results, out_path):
    curve_styles = {
        "baseline": {"label": "Baseline", "color": "#0072B2", "marker": "o"},
        "fgsm": {"label": "FGSM-trained", "color": "#009E73", "marker": "s"},
        "trades": {"label": "TRADES", "color": "#D55E00", "marker": "^"},
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for key, acc_dict in results.items():
        eps_values = sorted(acc_dict.keys())
        accuracies = [acc_dict[eps] for eps in eps_values]
        style = curve_styles.get(key, {})
        ax.plot(
            eps_values,
            accuracies,
            label=style.get("label", key),
            color=style.get("color", None),
            marker=style.get("marker", None),
            linestyle="-",
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


def save_results_to_csv(results: Dict[str, Dict[float, float]], csv_path: str) -> None:
    """Save results to a CSV with columns: regime, epsilon, accuracy."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["regime", "epsilon", "accuracy"]) 
        for regime, acc_dict in results.items():
            for eps, acc in acc_dict.items():
                writer.writerow([regime, float(eps), float(acc)])
    print(f"Saved results to {csv_path}")


def load_results_from_csv(csv_path: str) -> Dict[str, Dict[float, float]]:
    """Load results from a CSV created by save_results_to_csv."""
    results: Dict[str, Dict[float, float]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            regime = row["regime"]
            eps = float(row["epsilon"]) if row.get("epsilon") is not None else 0.0
            acc = float(row["accuracy"]) if row.get("accuracy") is not None else 0.0
            if regime not in results:
                results[regime] = {}
            results[regime][eps] = acc
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
        plot_results(results, plot_path)
        return

    results = run_experiment(max_steps=args.max_steps, log_every=args.log_every)

    # Print results table
    print("\nAccuracy results:", flush=True)
    for name, acc_dict in results.items():
        print(f"  {name}:", flush=True)
        for eps, acc in sorted(acc_dict.items()):
            print(f"    epsilon={eps:.1f}: accuracy={acc:.4f}", flush=True)

    # Save results and plot
    save_results_to_csv(results, args.results_csv)
    plot_path = os.path.join(args.output_dir, "accuracy_vs_epsilon.png")
    plot_results(results, plot_path)


if __name__ == "__main__":
    main()
