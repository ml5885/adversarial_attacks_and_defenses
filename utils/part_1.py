import os
import csv
from datetime import datetime, timezone
import numpy as np


def ensure_csv_with_header(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_id", "timestamp", "mode",
                "targeted", "norm", "loss_fn",
                "epsilon", "asr",
                "image_index", "eps_star",
                "n_eps"
            ])


def append_curve_rows(csv_path, run_id, targeted, norm, loss_fn, eps_grid, asr_list):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        ts = datetime.now(timezone.utc).isoformat()
        for eps, asr in zip(eps_grid, asr_list):
            writer.writerow([
                run_id, ts, "curve",
                int(targeted), norm, loss_fn,
                float(eps), float(asr),
                "", "",  # image_index, eps_star not used here
                ""       # n_eps
            ])


def append_eps_star_rows(csv_path, run_id, targeted, norm, loss_fn, eps_star_list):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        ts = datetime.now(timezone.utc).isoformat()
        for idx, eps_star in enumerate(eps_star_list):
            writer.writerow([
                run_id, ts, "eps_star",
                int(targeted), norm, loss_fn,
                "", "",         # epsilon, asr
                idx, float(eps_star),  # image_index, eps_star
                ""              # n_eps
            ])


def append_meta_row(csv_path, run_id, targeted, norm, loss_fn, n_eps):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        ts = datetime.now(timezone.utc).isoformat()
        writer.writerow([
            run_id, ts, "meta",
            int(targeted), norm, loss_fn,
            "", "",  # epsilon, asr
            "", "",  # image_index, eps_star
            int(n_eps)
        ])


def load_results_from_csv(csv_path):
    """Load curves and eps_star entries from CSV, grouped by condition."""
    curves = {}
    eps_stars = {}
    n_eps_meta = {}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row["mode"]
            targeted = bool(int(row["targeted"])) if row["targeted"] != "" else False
            norm = row["norm"]
            loss_fn = row["loss_fn"]
            key = (targeted, norm, loss_fn)

            if mode == "curve":
                eps = float(row["epsilon"])
                asr = float(row["asr"])
                curves.setdefault(key, ([], []))
                curves[key][0].append(eps)
                curves[key][1].append(asr)
            elif mode == "eps_star":
                if row["eps_star"] != "":
                    eps_star = float(row["eps_star"])
                    eps_stars.setdefault(key, [])
                    eps_stars[key].append(eps_star)
            elif mode == "meta":
                if row["n_eps"] != "":
                    n_eps_meta[key] = int(row["n_eps"])

    # Sort curves by epsilon
    for key in list(curves.keys()):
        eps, asr = curves[key]
        order = np.argsort(eps)
        curves[key] = (list(np.array(eps)[order]), list(np.array(asr)[order]))

    return curves, eps_stars, n_eps_meta
