import argparse
import csv
import os
import random
import requests
from datetime import datetime

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from tabulate import tabulate
from tqdm import tqdm

import attacks
from model import load_resnet18
import utils.part_1 as utils

NUM_IMAGES = 100
BATCH_SIZE = 10
OUTPUT_DIR = "results/part_1"
DEFAULT_CSV_PATH = os.path.join(OUTPUT_DIR, "results.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1000
PGD_STEPS = 40

BASE_EPS_LINF_GRID = np.linspace(0, 8 / 255, 9)  # {0, 1/255, ..., 8/255}
BASE_EPS_L2_GRID = np.linspace(0, 3.0, 10)       # equally spaced in [0, 3.0]

CW_KAPPA = 0.0


def get_imagenet_labels():
    """Downloads ImageNet class names."""
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        response.raise_for_status()
        return {i: name.strip() for i, name in enumerate(response.text.split("\n"))}
    except Exception as e:
        print(f"Warning: Could not download ImageNet labels: {e}. Class names will be numbers.")
        return {i: str(i) for i in range(NUM_CLASSES)}


def get_dataloader(checkpoint_path, num_images):
    """Samples and returns a DataLoader with <num_images> correctly classified
    images from the ImageNet validation set.
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])

    dataset = datasets.load_dataset("mrm8488/ImageNet1K-val", split="train", streaming=True)
    dataset = dataset.map(lambda x: {"image": transform(x["image"]), "label": x["label"]})
    
    print("Loading model for pre-filtering...")
    model = load_resnet18(checkpoint_path, DEVICE)

    print(f"Sampling {num_images} correctly classified images...")
    clean_images, clean_labels = [], []
    pbar = tqdm(total=num_images)

    for item in dataset:
        if len(clean_images) >= num_images:
            break
        image = item["image"].unsqueeze(0).to(DEVICE)
        label = torch.tensor([item["label"]]).to(DEVICE)

        with torch.no_grad():
            if model(image).argmax(dim=1).item() == label.item():
                clean_images.append(image)
                clean_labels.append(label)
                pbar.update(1)
    pbar.close()

    if len(clean_images) < num_images:
        print(f"Warning: Only found {len(clean_images)} correctly classified images.")

    images_tensor = torch.cat(clean_images)
    labels_tensor = torch.cat(clean_labels)

    target_labels = torch.tensor([
        (l.item() + random.randint(1, NUM_CLASSES - 1)) % NUM_CLASSES for l in labels_tensor
    ]).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(images_tensor, labels_tensor, target_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    del base_model, model
    torch.cuda.empty_cache()
    return loader


def plot_two_curves(eps1, asr1, label1, eps2, asr2, label2, title, filename):
    """Plot two ASR vs epsilon curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(eps1, asr1, marker="o", linestyle="-", label=label1, color="#0072B2")
    ax.plot(eps2, asr2, marker="s", linestyle="-", label=label2, color="#D55E00")

    ax.set_xlabel("Epsilon", fontsize=14)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(True, linewidth=0.5, linestyle="--", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def plot_example_image(clean_img, adv_img, clean_label_str, adv_label_str, title, filename):
    """Saves a plot of a clean, perturbed, and adversarial image."""
    clean_plot = clean_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    adv_plot = adv_img.cpu().squeeze(0).permute(1, 2, 0).numpy()

    perturbation = adv_plot - clean_plot
    # Normalize perturbation for visualization
    denom = (perturbation.max() - perturbation.min() + 1e-9)
    perturbation_viz = (perturbation - perturbation.min()) / denom

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.imshow(clean_plot)
    ax1.set_title(f"Clean Image\nPrediction: {clean_label_str}")
    ax1.axis("off")

    ax2.imshow(perturbation_viz)
    ax2.set_title("Perturbation (Normalized)")
    ax2.axis("off")

    ax3.imshow(adv_plot)
    ax3.set_title(f"Adversarial Image\nPrediction: {adv_label_str}")
    ax3.axis("off")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved example image: {filename}")


def _extend_grid_until_full_success(base_grid, norm_name, last_eps):
    """Given the fixed base grid and the last tried epsilon, return the next epsilon to try
    and the step size (keeps the same spacing as the base grid)."""
    if len(base_grid) >= 2:
        step = float(base_grid[1] - base_grid[0])
    else:
        step = (1.0 / 255.0) if norm_name == "linf" else 0.333333
    if last_eps is None:
        last_eps = float(base_grid[-1]) if len(base_grid) > 0 else 0.0
    next_eps = float(last_eps + step)
    return next_eps, step


def run_fixed_sweep(model, loader, norm, loss_fn, targeted, base_eps_grid):
    """Runs a sweep of attacks starting from the fixed epsilon grid.
    If ASR does not reach 100% by the last grid point, keep increasing epsilon
    by the same step size until full success is achieved.
    If ASR reaches 100% before the last grid point, end early.

    Returns:
        eps_grid (list): The grid of epsilons actually tested (possibly extended).
        asr_over_eps (list): The ASR at each epsilon.
        eps_star_per_image (list): The smallest epsilon that worked for each image.
        first_success (dict): Data for the first successful attack found.
        num_epsilons_used (int): How many epsilon points were needed to reach 100% ASR.
    """
    eps_grid = []
    asr_over_eps = []

    # Track the first epsilon that fools each image
    eps_star_map = {i: 0.0 for i in range(len(loader.dataset))}
    first_success_data = None

    # 1) Sweep the fixed grid (early stop if ASR reaches 100%)
    reached_full_success = False
    last_eps_tried = None

    for epsilon in base_eps_grid:
        epsilon = float(epsilon)
        last_eps_tried = epsilon

        successful_attacks_at_eps = 0
        total_images_processed = 0

        for i, (images, labels, target_labels) in enumerate(loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            target_labels = target_labels.to(DEVICE)

            adv_images = attacks.pgd(
                model=model,
                images=images,
                labels=labels,
                epsilon=epsilon,
                norm=norm,
                loss_fn=loss_fn,
                targeted=targeted,
                target_labels=target_labels if targeted else None,
                num_steps=PGD_STEPS,
                step_size=epsilon / 4.0,
                kappa=(CW_KAPPA if loss_fn == "cw" else 0.0),
                device=DEVICE,
            )

            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)

            for j in range(images.size(0)):
                img_idx = (i * loader.batch_size) + j
                true_label = labels[j]
                target_label = target_labels[j]
                pred = preds[j]

                if targeted:
                    is_success = (pred.item() == target_label.item())
                else:
                    is_success = (pred.item() != true_label.item())

                if is_success:
                    successful_attacks_at_eps += 1
                    if eps_star_map[img_idx] == 0.0 and epsilon > 0:
                        eps_star_map[img_idx] = epsilon
                        if first_success_data is None:
                            first_success_data = {
                                "clean_img": images[j].cpu(),
                                "adv_img": adv_images[j].cpu(),
                                "true_label_idx": true_label.item(),
                                "adv_label_idx": pred.item(),
                                "eps": epsilon,
                                "norm": norm,
                                "loss_fn": loss_fn,
                                "targeted": targeted,
                            }

            total_images_processed += images.size(0)

        asr = successful_attacks_at_eps / total_images_processed
        eps_grid.append(epsilon)
        asr_over_eps.append(asr)

        if asr == 1.0:
            print(f"  Reached 100% ASR at eps = {epsilon:.6f} within fixed grid")
            reached_full_success = True
            break  # early stop: no need to try larger fixed-grid epsilons

    # 2) If not yet at 100%, keep extending epsilon using the same step
    if not reached_full_success:
        print("  Fixed grid exhausted without 100% ASR; extending epsilon...")
        while True:
            next_eps, _ = _extend_grid_until_full_success(base_eps_grid, norm, last_eps_tried)
            last_eps_tried = next_eps

            successful_attacks_at_eps = 0
            total_images_processed = 0

            for i, (images, labels, target_labels) in enumerate(loader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                target_labels = target_labels.to(DEVICE)

                adv_images = attacks.pgd(
                    model=model,
                    images=images,
                    labels=labels,
                    epsilon=next_eps,
                    norm=norm,
                    loss_fn=loss_fn,
                    targeted=targeted,
                    target_labels=target_labels if targeted else None,
                    num_steps=PGD_STEPS,
                    step_size=(next_eps / 4.0),
                    kappa=(CW_KAPPA if loss_fn == "cw" else 0.0),
                    device=DEVICE,
                )

                with torch.no_grad():
                    preds = model(adv_images).argmax(dim=1)

                for j in range(images.size(0)):
                    img_idx = (i * loader.batch_size) + j
                    true_label = labels[j]
                    target_label = target_labels[j]
                    pred = preds[j]

                    if targeted:
                        is_success = (pred.item() == target_label.item())
                    else:
                        is_success = (pred.item() != true_label.item())

                    if is_success:
                        successful_attacks_at_eps += 1
                        if eps_star_map[img_idx] == 0.0 and next_eps > 0:
                            eps_star_map[img_idx] = next_eps
                            if first_success_data is None:
                                first_success_data = {
                                    "clean_img": images[j].cpu(),
                                    "adv_img": adv_images[j].cpu(),
                                    "true_label_idx": true_label.item(),
                                    "adv_label_idx": pred.item(),
                                    "eps": next_eps,
                                    "norm": norm,
                                    "loss_fn": loss_fn,
                                    "targeted": targeted,
                                }

                total_images_processed += images.size(0)

            asr = successful_attacks_at_eps / total_images_processed
            eps_grid.append(next_eps)
            asr_over_eps.append(asr)

            if asr == 1.0:
                print(f"  Reached 100% ASR at eps = {next_eps:.6f} after extending")
                break

    # Gather all epsilon* values (exclude zeros which indicate no success at positive eps)
    eps_star_list = [v for v in eps_star_map.values() if v > 0]

    return eps_grid, asr_over_eps, eps_star_list, first_success_data, len(eps_grid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", action="store_true", help="Skip experiments and load results from CSV to generate plots and tables.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to the CSV file to save/load results.")
    parser.add_argument("--ckpt", type=str, default="models/resnet18_l2_eps0.ckpt", help="Path to the ResNet-18 checkpoint file.")
    args = parser.parse_args()

    csv_path = args.csv
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    utils.ensure_csv_with_header(csv_path)

    if args.analysis:
        print("Analysis mode: loading results from CSV and generating outputs...")
        curves, eps_stars, n_eps_meta = utils.load_results_from_csv(csv_path)
        labels_map = get_imagenet_labels()

        # Build tables from eps_stars
        median_eps_star = {
            "untargeted": {"linf": {}, "l2": {}},
            "targeted": {"linf": {}, "l2": {}},
        }

        for targeted in [False, True]:
            for norm in ["linf", "l2"]:
                for loss_fn in ["ce", "cw"]:
                    key = (targeted, norm, loss_fn)
                    stars = eps_stars.get(key, [])
                    median_eps = np.median(stars) if len(stars) > 0 else np.inf
                    if targeted:
                        median_eps_star["targeted"][norm][loss_fn] = median_eps
                    else:
                        median_eps_star["untargeted"][norm][loss_fn] = median_eps

        # Print tables
        print("\n--- Median Epsilon* Tables (from CSV) ---")
        for targeted_str in ["untargeted", "targeted"]:
            table_data = []
            for norm in ["linf", "l2"]:
                ce_median = median_eps_star[targeted_str][norm].get("ce", np.inf)
                cw_median = median_eps_star[targeted_str][norm].get("cw", np.inf)
                norm_name = "L-inf" if norm == "linf" else "L2"
                table_data.append([
                    norm_name,
                    f"{ce_median:.6f}" if np.isfinite(ce_median) else "inf",
                    f"{cw_median:.6f}" if np.isfinite(cw_median) else "inf",
                ])
            headers = ["Norm", "Cross-Entropy (CE)", "Carlini-Wagner (CW)"]
            print(f"\n{targeted_str.capitalize()} Attacks (Median Epsilon*):")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Print number of epsilon points to reach 100% ASR (if present, else infer from curves)
        print("\n--- Number of epsilon points to reach 100% ASR (from CSV) ---")
        for targeted, norm, loss_fn in [
            (False, "linf", "ce"), (False, "linf", "cw"),
            (False, "l2",   "ce"), (False, "l2",   "cw"),
            (True,  "linf", "ce"), (True,  "linf", "cw"),
            (True,  "l2",   "ce"), (True,  "l2",   "cw"),
        ]:
            key = (targeted, norm, loss_fn)
            n = n_eps_meta.get(key)
            if n is None and key in curves:
                eps, asr = curves[key]
                n = len(eps)
                # Optional: find first index where ASR hits 1.0
                for idx, val in enumerate(asr):
                    if val == 1.0:
                        n = idx + 1
                        break
            if n is not None:
                tstr = "Targeted" if targeted else "Untargeted"
                nstr = "L-inf" if norm == "linf" else "L2"
                print(f"{tstr}, {nstr}, {loss_fn.upper()}: {n} epsilon points")

        # Generate plots from curves
        print("\n--- Generating ASR vs. Epsilon Plots (from CSV) ---")
        for targeted, norm in [(False, "linf"), (False, "l2"), (True, "linf"), (True, "l2")]:
            key_ce = (targeted, norm, "ce")
            key_cw = (targeted, norm, "cw")
            if key_ce not in curves or key_cw not in curves:
                print(f"Skipping plot for {(targeted, norm)} because one or both curves are missing.")
                continue

            ce_eps, ce_asr = curves[key_ce]
            cw_eps, cw_asr = curves[key_cw]

            tlabel = "Targeted" if targeted else "Untargeted"
            norm_label = "L-inf" if norm == "linf" else "L2"
            title = f"Success Rate vs. Epsilon ({tlabel}, {norm_label})"
            filename = os.path.join(OUTPUT_DIR, f"asr_{tlabel.lower()}_{norm}.png")

            plot_two_curves(ce_eps, ce_asr, "Cross-Entropy (CE)", cw_eps, cw_asr, "Carlini-Wagner (CW)", title, filename)

        print("\nAnalysis complete. Plots and tables generated from CSV.")
        return

    # Experiment mode
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading ResNet-18 model from local checkpoint...")
    model = load_resnet18(args.ckpt, DEVICE)

    loader = get_dataloader(args.ckpt, args.NUM_IMAGES)
    labels_map = get_imagenet_labels()

    # Store raw curves for plotting directly
    curves = {}  # key: (targeted, norm, loss_fn) -> (eps_grid, asr_list)
    median_eps_star = {
        "untargeted": {"linf": {}, "l2": {}},
        "targeted": {"linf": {}, "l2": {}},
    }
    eps_point_counts = {}
    first_successful_attack = None

    attack_configs = [
        # (targeted, norm, base_grid)
        (False, "linf", BASE_EPS_LINF_GRID),
        (False, "l2",   BASE_EPS_L2_GRID),
        (True,  "linf", BASE_EPS_LINF_GRID),
        (True,  "l2",   BASE_EPS_L2_GRID),
    ]

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    for targeted, norm, base_grid in attack_configs:
        for loss_fn in ["ce", "cw"]:
            target_str = "Targeted" if targeted else "Untargeted"
            print(f"\nRunning sweep: {target_str}, {norm.upper()}, {loss_fn.upper()} Loss...")

            eps_grid, asr_list, eps_star_list, first_success, n_eps = run_fixed_sweep(
                model, loader, norm, loss_fn, targeted, base_grid
            )

            # Track curves
            curves[(targeted, norm, loss_fn)] = (eps_grid, asr_list)
            eps_point_counts[(targeted, norm, loss_fn)] = n_eps

            # For tables (median epsilon*)
            median_eps = np.median(eps_star_list) if eps_star_list else np.inf
            if targeted:
                median_eps_star["targeted"][norm][loss_fn] = median_eps
            else:
                median_eps_star["untargeted"][norm][loss_fn] = median_eps

            # Save to CSV
            utils.append_curve_rows(csv_path, run_id, targeted, norm, loss_fn, eps_grid, asr_list)
            utils.append_eps_star_rows(csv_path, run_id, targeted, norm, loss_fn, eps_star_list)
            utils.append_meta_row(csv_path, run_id, targeted, norm, loss_fn, n_eps)

            # Example image
            if first_success and not first_successful_attack:
                first_successful_attack = first_success

    # Tables
    print("\n--- Median Epsilon* Tables ---")
    for targeted_str in ["untargeted", "targeted"]:
        table_data = []
        for norm in ["linf", "l2"]:
            ce_median = median_eps_star[targeted_str][norm].get("ce", np.inf)
            cw_median = median_eps_star[targeted_str][norm].get("cw", np.inf)
            norm_name = "L-inf" if norm == "linf" else "L2"
            table_data.append([
                norm_name,
                f"{ce_median:.6f}" if np.isfinite(ce_median) else "inf",
                f"{cw_median:.6f}" if np.isfinite(cw_median) else "inf",
            ])

        headers = ["Norm", "Cross-Entropy (CE)", "Carlini-Wagner (CW)"]
        print(f"\n{targeted_str.capitalize()} Attacks (Median Epsilon*):")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Number of epsilon points
    print("\n--- Number of epsilon points to reach 100% ASR ---")
    for targeted, norm, loss_fn in [
        (False, "linf", "ce"), (False, "linf", "cw"),
        (False, "l2",   "ce"), (False, "l2",   "cw"),
        (True,  "linf", "ce"), (True,  "linf", "cw"),
        (True,  "l2",   "ce"), (True,  "l2",   "cw"),
    ]:
        n = eps_point_counts.get((targeted, norm, loss_fn), None)
        if n is not None:
            tstr = "Targeted" if targeted else "Untargeted"
            nstr = "L-inf" if norm == "linf" else "L2"
            print(f"{tstr}, {nstr}, {loss_fn.upper()}: {n} epsilon points")

    # Plots
    print("\n--- Generating ASR vs. Epsilon Plots ---")
    for (targeted, norm), _ in { (False, "linf"): None, (False, "l2"): None, (True, "linf"): None, (True, "l2"): None }.items():
        ce_eps, ce_asr = curves[(targeted, norm, "ce")]
        cw_eps, cw_asr = curves[(targeted, norm, "cw")]

        tlabel = "Targeted" if targeted else "Untargeted"
        norm_label = "L-inf" if norm == "linf" else "L2"
        title = f"Success Rate vs. Epsilon ({tlabel}, {norm_label})"
        filename = os.path.join(OUTPUT_DIR, f"asr_{tlabel.lower()}_{norm}.png")

        plot_two_curves(ce_eps, ce_asr, "Cross-Entropy (CE)", cw_eps, cw_asr, "Carlini-Wagner (CW)", title, filename)

    # Example image
    print("\n--- Generating Example Attacked Image ---")
    if first_successful_attack:
        info = first_successful_attack

        # Move back to device for inference
        clean_img_tensor = info["clean_img"].unsqueeze(0).to(DEVICE)
        adv_img_tensor = info["adv_img"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            clean_logits = model(clean_img_tensor)
            clean_probs = F.softmax(clean_logits, dim=1)
            clean_conf, clean_idx = clean_probs.max(dim=1)
            clean_label_name = labels_map.get(clean_idx.item(), clean_idx.item())
            clean_label_str = f"{clean_label_name} ({clean_conf.item() * 100:.1f}%)"

            adv_logits = model(adv_img_tensor)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_conf, adv_idx = adv_probs.max(dim=1)
            adv_label_name = labels_map.get(adv_idx.item(), adv_idx.item())
            adv_label_str = f"{adv_label_name} ({adv_conf.item() * 100:.1f}%)"

        eps_str = f"{info['eps']:.6f}"
        norm_str = info["norm"].upper()
        loss_str = info["loss_fn"].upper()
        target_str = "Targeted" if info["targeted"] else "Untargeted"

        title = f"Example Attack ({target_str})\nNorm: {norm_str}, Loss: {loss_str}, Epsilon* = {eps_str}"
        filename = os.path.join(OUTPUT_DIR, "example_attack.png")

        plot_example_image(
            info["clean_img"], info["adv_img"],
            clean_label_str, adv_label_str,
            title, filename
        )
    else:
        print("Could not generate example image as no attacks were successful.")

    print(f"\nAll experiments complete. Results and CSV are in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
