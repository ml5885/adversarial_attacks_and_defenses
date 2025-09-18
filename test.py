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

# Standard ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class NormalizedModel(torch.nn.Module):
    """
    A wrapper for a standard torchvision model.
    This wrapper adds the normalization step inside the forward pass,
    so the attack functions can work with images in the [0, 1] range.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Create a non-learnable buffer for normalization
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        # Move mean and std to the correct device
        device = x.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        
        # Normalize the input tensor (x is assumed to be in [0, 1])
        x_normalized = (x - mean) / std
        # Pass the normalized input to the underlying model
        return self.model(x_normalized)


def format_title(attack, targeted, norm, loss, epsilon, target_label=None):
    norm = norm.lower()
    loss = loss.lower()
    eps_fmt = f"{epsilon:.3f}" if norm == 'linf' else f"{epsilon:.2f}"  # finer for linf
    base = f"{attack} {'targeted' if targeted else 'untargeted'} | {norm} | {loss} | eps={eps_fmt}"
    if targeted and target_label:
        base += f" -> {target_label}"
    return base

def get_imagenet_labels():
    """Downloads and returns a mapping of ImageNet class IDs to names."""
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        labels_map = {i: name.strip() for i, name in enumerate(response.text.split('\n'))}
        return labels_map
    except requests.RequestException as e:
        print(f"Error downloading ImageNet labels: {e}")
        return None

def preprocess_image(img_url):
    """Downloads an image and preprocesses it to a [0, 1] tensor."""
    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Preprocessing to a [0, 1] tensor
        preprocess = T.Compose([
            T.Resize(256),  
            T.CenterCrop(224),  
            T.ToTensor(),  
        ])
        return preprocess(img).unsqueeze(0)  # Add batch dimension
    except requests.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_prediction(model, image, labels_map):
    """Gets the top-1 prediction for a given model and image."""
    with torch.no_grad():
        logits = model(image)  # Image is [0, 1], model is NormalizedModel
        probabilities = F.softmax(logits, dim=1)
        top1_prob, top1_idx = torch.topk(probabilities, 1)
        pred_idx = top1_idx.item()
        pred_label = labels_map.get(pred_idx, "Unknown")
        pred_prob = top1_prob.item() * 100
        return f"{pred_label} ({pred_prob:.1f}%)", pred_idx

def plot_attack(clean_img, adv_img, clean_label, adv_label, title, filename):
    """Plots the clean and adversarial images side-by-side and saves to disk."""
    clean_plot = clean_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    adv_plot = adv_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    perturbation = (adv_plot - clean_plot)
    perturbation_viz = perturbation - perturbation.min()
    perturbation_viz /= (perturbation_viz.max() + 1e-12)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(clean_plot)
    ax1.set_title(f"Clean Image\nPrediction: {clean_label}")
    ax1.axis('off')
    
    ax2.imshow(perturbation_viz)
    ax2.set_title("Perturbation (Scaled)")
    ax2.axis('off')

    ax3.imshow(adv_plot)
    ax3.set_title(f"Adversarial Image\nPrediction: {adv_label}")
    ax3.axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to: {filename}")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a directory for plots if it doesn't exist
    plot_dir = "attack_examples"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to '{plot_dir}/'")

    # 1. Load ImageNet labels
    labels_map = get_imagenet_labels()
    if labels_map is None:
        return

    # 2. Load and preprocess a sample image (Golden Retriever)
    # img_url = "https://salient-imagenet.cs.umd.edu/feature_visualization/class_388/feature_1437/images/3.jpg"
    img_url = "https://petzpark.com.au/cdn/shop/articles/Breeds_Thumbnails_4_1_833358e4-eda0-43b3-898f-297e33980ab4_900x.jpg"
    clean_image_tensor = preprocess_image(img_url) # This is in [0, 1]
    if clean_image_tensor is None:
        return
    clean_image_tensor = clean_image_tensor.to(device)

    # 3. Load model from local checkpoint
    print("Loading ResNet-18 model...")

    # Load the base model (which expects normalized inputs)
    base_model = torchvision.models.resnet18(weights=None)
    
    checkpoint_path = "models/resnet18_l2_eps0.ckpt"
    try:
        print(f"Loading state dict from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if 'model' in checkpoint and isinstance(checkpoint['model'], dict):
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        module_prefix = "module."
        model_prefix = "model."
        
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith(module_prefix):
                new_k = new_k[len(module_prefix):]
            if new_k.startswith(model_prefix):
                new_k = new_k[len(model_prefix):]
            cleaned_state_dict[new_k] = v

        base_model.load_state_dict(cleaned_state_dict, strict=False)
        
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        print("Please make sure 'resnet18_l2_eps0.ckpt' is in the same directory.")
        return
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("The checkpoint file might be corrupt or incompatible.")
        return
    
    # Wrap the base model with the NormalizedModel wrapper
    # All attacks and predictions will use this 'model' object
    model = NormalizedModel(base_model).to(device).eval()
    
    # 4. Get clean prediction
    # This now works: get_prediction feeds [0,1] image to the
    # wrapped 'model', which handles normalization.
    clean_pred_label, clean_pred_idx = get_prediction(model, clean_image_tensor, labels_map)
    print(f"Clean prediction: {clean_pred_label}")
    labels_tensor = torch.tensor([clean_pred_idx], device=device)

    # --- Attack Parameters ---
    epsilon_linf = 8/255.0  # L-infinity budget
    epsilon_l2 = 3.0         # L-2 budget
    pgd_steps = 40           # Standard PGD steps
    
    target_label_idx = 963 # "pizza"
    target_label_name = labels_map.get(target_label_idx, "Unknown")
    target_labels_tensor = torch.tensor([target_label_idx], device=device)
    print(f"Targeting class: {target_label_name} (ID: {target_label_idx})")

    # --- Run All 8 Attack Combinations ---
    # All attack calls now correctly pass the wrapped 'model'
    
    # --- 1. Untargeted, L-infinity, CE ---
    print("\nRunning PGD (Untargeted, L-inf, CE)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_linf, norm="linf", loss_fn="ce",  
        targeted=False, num_steps=pgd_steps, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", False, "linf", "ce", epsilon_linf)
    filename = os.path.join(plot_dir, "attack_pgd_linf_untargeted_ce.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    # --- 2. Untargeted, L-infinity, CW ---
    print("\nRunning PGD (Untargeted, L-inf, CW)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_linf, norm="linf", loss_fn="cw",  
        targeted=False, num_steps=pgd_steps, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", False, "linf", "cw", epsilon_linf)
    filename = os.path.join(plot_dir, "attack_pgd_linf_untargeted_cw.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    # --- 3. Untargeted, L-2, CE ---
    print("\nRunning PGD (Untargeted, L-2, CE)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_l2, norm="l2", loss_fn="ce",  
        targeted=False, num_steps=pgd_steps, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", False, "l2", "ce", epsilon_l2)
    filename = os.path.join(plot_dir, "attack_pgd_l2_untargeted_ce.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    # --- 4. Untargeted, L-2, CW ---
    print("\nRunning PGD (Untargeted, L-2, CW)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_l2, norm="l2", loss_fn="cw",  
        targeted=False, num_steps=pgd_steps, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", False, "l2", "cw", epsilon_l2)
    filename = os.path.join(plot_dir, "attack_pgd_l2_untargeted_cw.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    # --- 5. Targeted, L-infinity, CE ---
    print("\nRunning PGD (Targeted, L-inf, CE)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_linf, norm="linf", loss_fn="ce",  
        targeted=True, target_labels=target_labels_tensor,
        num_steps=pgd_steps, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", True, "linf", "ce", epsilon_linf, target_label_name)
    filename = os.path.join(plot_dir, "attack_pgd_linf_targeted_ce.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    # --- 6. Targeted, L-infinity, CW ---
    print("\nRunning PGD (Targeted, L-inf, CW)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_linf, norm="linf", loss_fn="cw",  
        targeted=True, target_labels=target_labels_tensor,
        num_steps=pgd_steps, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", True, "linf", "cw", epsilon_linf, target_label_name)
    filename = os.path.join(plot_dir, "attack_pgd_linf_targeted_cw.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    # --- 7. Targeted, L-2, CE ---
    print("\nRunning PGD (Targeted, L-2, CE)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_l2, norm="l2", loss_fn="ce",  
        targeted=True, target_labels=target_labels_tensor,
        num_steps=pgd_steps * 2, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", True, "l2", "ce", epsilon_l2, target_label_name)
    filename = os.path.join(plot_dir, "attack_pgd_l2_targeted_ce.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    # --- 8. Targeted, L-2, CW ---
    print("\nRunning PGD (Targeted, L-2, CW)...")
    adv_img = attacks.pgd(
        model, clean_image_tensor, labels_tensor,
        epsilon=epsilon_l2, norm="l2", loss_fn="cw",  
        targeted=True, target_labels=target_labels_tensor,
        num_steps=pgd_steps * 2, device=device
    )
    adv_label, _ = get_prediction(model, adv_img, labels_map)
    title = format_title("PGD", True, "l2", "cw", epsilon_l2, target_label_name)
    filename = os.path.join(plot_dir, "attack_pgd_l2_targeted_cw.png")
    plot_attack(clean_image_tensor, adv_img, clean_pred_label, adv_label, title, filename)

    print(f"\nAll 8 attack examples generated and plots saved to '{plot_dir}/'")

if __name__ == "__main__":
    main()