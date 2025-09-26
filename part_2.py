import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import torch

import train
import model


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run_experiment(max_steps):
    """Train baseline, FGSM and TRADES models and evaluate them.

    Args:
        max_steps: Maximum number of optimisation steps for each
            training routine.

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
    print("Training baseline model...")
    baseline_model = model.ConvNet()
    baseline_model = train.train_baseline(
        baseline_model,
        train_loader,
        device=device,
        lr=1e-4,
        max_steps=max_steps,
    )
    results["baseline"] = train.evaluate_fgsm(
        baseline_model, test_loader, device=device, epsilons=eval_eps
    )

    # FGSM adversarial training
    print("Training FGSM adversarially trained model...")
    fgsm_model = model.ConvNet()
    fgsm_model = train.train_fgsm(
        fgsm_model,
        train_loader,
        device=device,
        epsilon_train=0.3,
        lr=1e-4,
        max_steps=max_steps,
    )
    results["fgsm"] = train.evaluate_fgsm(
        fgsm_model, test_loader, device=device, epsilons=eval_eps
    )

    # TRADES training
    print("Training TRADES adversarially trained model...")
    trades_model = model.ConvNet()
    trades_model = train.train_trades(
        trades_model,
        train_loader,
        device=device,
        epsilon_train=0.3,
        beta=6.0,
        lr=1e-4,
        max_steps=max_steps,
    )
    results["trades"] = train.evaluate_fgsm(
        trades_model, test_loader, device=device, epsilons=eval_eps
    )

    return results


def plot_results(results: Dict[str, Dict[float, float]], out_path: str) -> None:
    """Plot accuracy versus epsilon for multiple training regimes.

    Args:
        results: A nested dictionary as returned by run_experiment.
        out_path: Filepath at which to save the resulting PNG.
    """
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
    ax.set_xlabel("Epsilon", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_title("MNIST Adversarial Training: Accuracy vs. FGSM Perturbation", fontsize=16)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0.0, max(max(k for acc_dict in results.values() for k in acc_dict.keys()), 0.3))
    ax.grid(True, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adversarial training experiments on MNIST.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100_000,
        help="Maximum number of optimisation steps for each training regime (default: 100000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join("results", "part_2"),
        help="Directory in which to save results and plots.",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    results = run_experiment(max_steps=args.max_steps)
    # Also print a simple table to the console
    print("\nAccuracy results:")
    for name, acc_dict in results.items():
        print(f"  {name}:")
        for eps, acc in sorted(acc_dict.items()):
            print(f"    epsilon={eps:.1f}: accuracy={acc:.4f}")
    # Plot and save
    plot_path = os.path.join(args.output_dir, "accuracy_vs_epsilon.png")
    plot_results(results, plot_path)


if __name__ == "__main__":
    main()
