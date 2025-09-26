import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import requests
import matplotlib.pyplot as plt
import io
import torch.nn.functional as F
import os

import matplotlib
matplotlib.rcParams.update({
    'font.family': "serif",
    'axes.titlesize': 14,
    'figure.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

import attacks
from model import load_resnet18

CW_KAPPA = 50.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean.to(x.device)) / self.std.to(x.device))


def format_title(attack, targeted, norm, loss, epsilon, target_label=None):
    norm = norm.lower()
    loss = loss.lower()
    eps_fmt = f"{epsilon:.3f}" if norm == 'linf' else f"{epsilon:.2f}"
    base = f"{attack} {'targeted' if targeted else 'untargeted'} | {norm} | {loss} | eps={eps_fmt}"
    if targeted and target_label:
        base += f" | target = {target_label}"
    return base


def get_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        r = requests.get(url)
        r.raise_for_status()
        return {i: name.strip() for i, name in enumerate(r.text.split("\n"))}
    except requests.RequestException as e:
        print(f"Error downloading ImageNet labels: {e}")
        return None


def preprocess_image(img_url):
    """Load image from URL and convert to tensor."""
    try:
        r = requests.get(img_url, stream=True)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        preprocess = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        return preprocess(img).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def get_prediction(model, image, labels_map):
    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)
        p, idx = torch.topk(probs, 1)
        return f"{labels_map.get(idx.item(), 'Unknown')} ({p.item() * 100:.1f}%)", idx.item()


def plot_attack(clean_img, adv_img, clean_label, adv_label, title, filename):
    # Prepare images for plotting
    def to_numpy(img):
        img = img.detach().cpu().squeeze(0)
        img = img.permute(1, 2, 0).numpy()
        img = img.clip(0, 1)
        return img

    clean_np = to_numpy(clean_img)
    adv_np = to_numpy(adv_img)
    pert = adv_np - clean_np

    # Scale perturbation for visualization
    pert_viz = pert - pert.min()
    pert_viz /= pert_viz.max() + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax_clean, ax_pert, ax_adv = axes

    ax_clean.imshow(clean_np)
    ax_clean.set_title(f"Clean\n{clean_label}")
    ax_clean.axis('off')

    ax_pert.imshow(pert_viz)
    ax_pert.set_title("Perturbation (scaled)")
    ax_pert.axis('off')

    ax_adv.imshow(adv_np)
    ax_adv.set_title(f"Adversarial\n{adv_label}")
    ax_adv.axis('off')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir = "results/attack_examples3"
    os.makedirs(plot_dir, exist_ok=True)

    labels_map = get_imagenet_labels()
    if labels_map is None:
        return

    img_url = "https://petzpark.com.au/cdn/shop/articles/Breeds_Thumbnails_4_1_833358e4-eda0-43b3-898f-297e33980ab4_900x.jpg"
    x_clean = preprocess_image(img_url)
    if x_clean is None:
        return
    x_clean = x_clean.to(device)

    # Load model
    ckpt_path = "models/resnet18_l2_eps0.ckpt"
    model = load_resnet18(ckpt_path, device)

    clean_label, clean_idx = get_prediction(model, x_clean, labels_map)
    y = torch.tensor([clean_idx], device=device)

    eps_linf = 8 / 255.0
    eps_l2 = 3.0
    steps = 40

    target_idx = 963  # pizza
    target_name = labels_map.get(target_idx, "Unknown")
    t = torch.tensor([target_idx], device=device)
    print(f"Target label for targeted runs: {target_name} ({target_idx})")

    # 1) Untargeted, L-inf, CE
    adv = attacks.pgd(model, x_clean, y, eps_linf, norm="linf", loss_fn="ce",
                      targeted=False, num_steps=steps, device=device)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD untargeted | linf | ce | eps={eps_linf:.3f}",
                os.path.join(plot_dir, "pgd_linf_untargeted_ce.png"))

    # 2) Untargeted, L-inf, CW
    adv = attacks.pgd(model, x_clean, y, eps_linf, norm="linf", loss_fn="cw",
                      targeted=False, num_steps=steps, device=device, kappa=CW_KAPPA)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD untargeted | linf | cw | eps={eps_linf:.3f}",
                os.path.join(plot_dir, "pgd_linf_untargeted_cw.png"))

    # 3) Untargeted, L2, CE
    adv = attacks.pgd(model, x_clean, y, eps_l2, norm="l2", loss_fn="ce",
                      targeted=False, num_steps=steps, device=device)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD untargeted | l2 | ce | eps={eps_l2:.2f}",
                os.path.join(plot_dir, "pgd_l2_untargeted_ce.png"))

    # 4) Untargeted, L2, CW
    adv = attacks.pgd(model, x_clean, y, eps_l2, norm="l2", loss_fn="cw",
                      targeted=False, num_steps=steps * 2, device=device, kappa=CW_KAPPA)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD untargeted | l2 | cw | eps={eps_l2:.2f}",
                os.path.join(plot_dir, "pgd_l2_untargeted_cw.png"))

    # 5) Targeted, L-inf, CE
    adv = attacks.pgd(model, x_clean, y, eps_linf, norm="linf", loss_fn="ce",
                      targeted=True, target_labels=t, num_steps=steps, device=device)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD targeted | linf | ce | eps={eps_linf:.3f} | target = {target_name}",
                os.path.join(plot_dir, "pgd_linf_targeted_ce.png"))

    # 6) Targeted, L-inf, CW
    adv = attacks.pgd(model, x_clean, y, eps_linf, norm="linf", loss_fn="cw",
                      targeted=True, target_labels=t, num_steps=steps, device=device, kappa=CW_KAPPA)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD targeted | linf | cw | eps={eps_linf:.3f} | target = {target_name}",
                os.path.join(plot_dir, "pgd_linf_targeted_cw.png"))

    # 7) Targeted, L2, CE
    adv = attacks.pgd(model, x_clean, y, eps_l2, norm="l2", loss_fn="ce",
                      targeted=True, target_labels=t, num_steps=steps * 2, device=device)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD targeted | l2 | ce | eps={eps_l2:.2f} | target = {target_name}",
                os.path.join(plot_dir, "pgd_l2_targeted_ce.png"))

    # 8) Targeted, L2, CW
    adv = attacks.pgd(model, x_clean, y, eps_l2, norm="l2", loss_fn="cw",
                      targeted=True, target_labels=t, num_steps=steps * 2, device=device, kappa=CW_KAPPA)
    adv_label, _ = get_prediction(model, adv, labels_map)
    plot_attack(x_clean, adv, clean_label, adv_label,
                f"PGD targeted | l2 | cw | eps={eps_l2:.2f} | target = {target_name}",
                os.path.join(plot_dir, "pgd_l2_targeted_cw.png"))

    print(f"All 8 attack examples saved to '{plot_dir}/'")


if __name__ == "__main__":
    main()