import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
import os
import random
import requests

import attacks

NUM_IMAGES = 100
BATCH_SIZE = 10 
OUTPUT_DIR = "results/part1_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1000
PGD_STEPS = 40

BASE_EPS_LINF_GRID = np.linspace(0, 8/255, 9)
BASE_EPS_L2_GRID = np.linspace(0, 3.0, 10) 


class NormalizedModel(torch.nn.Module):
    """
    Wraps a model to include ImageNet normalization, so attacks can
    work on images in the [0, 1] range.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Register buffers so they are moved to the correct device
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x_normalized = (x - self.mean) / self.std
        return self.model(x_normalized)

def get_imagenet_labels():
    """Downloads ImageNet class names."""
    try:
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        response = requests.get(url)
        response.raise_for_status()
        return {i: name.strip() for i, name in enumerate(response.text.split('\n'))}
    except Exception as e:
        print(f"Warning: Could not download ImageNet labels: {e}. Class names will be numbers.")
        return {i: str(i) for i in range(NUM_CLASSES)}

def get_dataloader(num_images):
    """
    Samples and returns a DataLoader with <num_images> *correctly classified*
    images from the ImageNet validation set.
    """
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize(256), 
        T.CenterCrop(224), 
        T.ToTensor()
    ])
    
    dataset = datasets.load_dataset("mrm8488/ImageNet1K-val", split="train", streaming=True)
    dataset = dataset.map(lambda x: {'image': transform(x['image']), 'label': x['label']})
    
    print("Loading model for pre-filtering...")
    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model = NormalizedModel(base_model).to(DEVICE).eval()
    
    print(f"Sampling {num_images} correctly classified images...")
    clean_images, clean_labels = [], []
    pbar = tqdm(total=num_images)
    
    for item in dataset:
        if len(clean_images) >= num_images:
            break
        image = item['image'].unsqueeze(0).to(DEVICE)
        label = torch.tensor([item['label']]).to(DEVICE)
        
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

def plot_asr_curves(eps_grid, ce_asr, cw_asr, title, filename):
    """Saves a plot of ASR vs. Epsilon for CE and CW losses."""
    plt.rcParams.update({'font.family': 'serif'})
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(eps_grid, ce_asr, 'o-', label='Cross-Entropy Loss', color='royalblue')
    ax.plot(eps_grid, cw_asr, 's-', label='Carlini-Wagner Loss', color='firebrick')
    
    ax.set_xlabel('Epsilon', fontsize=12)
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=15)
    ax.legend(fontsize=11)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(left=0)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot: {filename}")
    
def plot_example_image(clean_img, adv_img, clean_label_str, adv_label_str, title, filename):
    """Saves a plot of a clean, perturbed, and adversarial image."""
    clean_plot = clean_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    adv_plot = adv_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    perturbation = adv_plot - clean_plot
    # Normalize perturbation for visualization
    perturbation_viz = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min() + 1e-9)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    ax1.imshow(clean_plot)
    ax1.set_title(f"Clean Image\nPrediction: {clean_label_str}")
    ax1.axis('off')
    
    ax2.imshow(perturbation_viz)
    ax2.set_title("Perturbation (Normalized)")
    ax2.axis('off')

    ax3.imshow(adv_plot)
    ax3.set_title(f"Adversarial Image\nPrediction: {adv_label_str}")
    ax3.axis('off')
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved example image: {filename}")

def run_fixed_sweep(model, loader, norm, loss_fn, targeted, base_eps_grid):
    """
    Runs a fixed sweep of attacks for a given configuration.
    
    Returns:
        eps_grid (list): The grid of epsilons used (may be extended).
        asr_over_eps (list): The ASR at each epsilon.
        eps_star_per_image (list): The smallest epsilon that worked for each image.
        first_success (dict): Data for the first successful attack found.
    """
    eps_grid = list(base_eps_grid)
    asr_over_eps = []
    
    # Store results for each image (by its index)
    # 0 = not yet successful, >0 = eps_star
    eps_star_map = {i: 0.0 for i in range(len(loader.dataset))}
    first_success_data = None
    
    current_eps_idx = 0
    
    while True:
        if current_eps_idx >= len(eps_grid):
            # Extend the grid
            if norm == 'linf':
                next_eps = eps_grid[-1] + 1/255.0
            else:
                next_eps = eps_grid[-1] + 0.5
            eps_grid.append(next_eps)
            print(f"  ASR < 100%, extending sweep to eps = {next_eps:.4f}")
        
        epsilon = eps_grid[current_eps_idx]
        
        successful_attacks_at_eps = 0
        total_images_processed = 0
        
        for i, (images, labels, target_labels) in enumerate(loader):
            images, labels, target_labels = images.to(DEVICE), labels.to(DEVICE), target_labels.to(DEVICE)
            
            # 1. Run attack
            adv_images = attacks.pgd(
                model, images, labels, epsilon,
                norm=norm, loss_fn=loss_fn, targeted=targeted,
                target_labels=target_labels if targeted else None,
                num_steps=PGD_STEPS, device=DEVICE,
                step_size=epsilon / 4
            )
            
            # 2. Check predictions
            with torch.no_grad():
                preds = model(adv_images).argmax(dim=1)
            
            # 3. Log results
            for j in range(images.size(0)):
                img_idx = (i * loader.batch_size) + j
                true_label = labels[j]
                target_label = target_labels[j]
                pred = preds[j]
                
                is_success = False
                if targeted:
                    is_success = (pred.item() == target_label.item())
                else:
                    is_success = (pred.item() != true_label.item())
                
                if is_success:
                    successful_attacks_at_eps += 1
                    
                    # If this is the *first* time this image was fooled,
                    # record this epsilon as its eps_star
                    if eps_star_map[img_idx] == 0.0 and epsilon > 0:
                        eps_star_map[img_idx] = epsilon
                        
                        # Save data for the example plot
                        if first_success_data is None:
                            first_success_data = {
                                'clean_img': images[j].cpu(),
                                'adv_img': adv_images[j].cpu(),
                                'true_label_idx': true_label.item(),
                                'adv_label_idx': pred.item(),
                                'eps': epsilon, 'norm': norm, 
                                'loss_fn': loss_fn, 'targeted': targeted
                            }
                            
            total_images_processed += images.size(0)
        
        # Calculate ASR for this epsilon
        asr = successful_attacks_at_eps / total_images_processed
        asr_over_eps.append(asr)
        
        # Check stop condition: only break if ASR reaches 100%
        if asr == 1.0:
            print(f"  Reached 100% ASR at eps = {epsilon:.4f}")
            break
        
        current_eps_idx += 1
        
    # Collect all found eps_star values (excluding those that never failed, value 0)
    eps_star_list = [v for v in eps_star_map.values() if v > 0]
    
    # Only return the part of the eps_grid that was actually tested
    tested_eps_grid = eps_grid[:len(asr_over_eps)]
    return tested_eps_grid, asr_over_eps, eps_star_list, first_success_data


def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Model
    print("Loading ResNet-18 model...")
    base_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model = NormalizedModel(base_model).to(DEVICE).eval()
    
    # 2. Load Data
    loader = get_dataloader(NUM_IMAGES)
    labels_map = get_imagenet_labels()
    
    # 3. Run all sweeps
    # Results containers
    plot_data = {}  # For ASR vs. eps plots
    median_eps_star = {
        'untargeted': {'linf': {}, 'l2': {}},
        'targeted': {'linf': {}, 'l2': {}}
    }
    first_successful_attack = None
    
    attack_configs = [
        # (targeted, norm, base_grid)
        (False, 'linf', BASE_EPS_LINF_GRID),
        (False, 'l2',   BASE_EPS_L2_GRID),
        (True,  'linf', BASE_EPS_LINF_GRID),
        (True,  'l2',   BASE_EPS_L2_GRID)
    ]
    
    for targeted, norm, base_grid in attack_configs:
        for loss_fn in ['ce', 'cw']:
            target_str = "Targeted" if targeted else "Untargeted"
            print(f"\nRunning sweep: {target_str}, {norm.upper()}, {loss_fn.upper()} Loss...")
            
            eps_grid, asr_list, eps_star_list, first_success = run_fixed_sweep(
                model, loader, norm, loss_fn, targeted, base_grid
            )
            
            # Store data for plots
            plot_key = (targeted, norm)
            if plot_key not in plot_data:
                plot_data[plot_key] = {}
            plot_data[plot_key][loss_fn] = (eps_grid, asr_list)
            
            # Store data for tables
            if eps_star_list:
                median_eps = np.median(eps_star_list)
            else:
                median_eps = np.inf # No attacks succeeded
            median_eps_star[target_str.lower()][norm][loss_fn] = median_eps
            
            # Store example image data
            if first_success and not first_successful_attack:
                first_successful_attack = first_success
                
    # 4. Generate and Print Tables (Deliverable 1)
    print("\n--- Median Epsilon* Tables ---")
    for targeted_str in ['untargeted', 'targeted']:
        table_data = []
        for norm in ['linf', 'l2']:
            ce_median = median_eps_star[targeted_str][norm]['ce']
            cw_median = median_eps_star[targeted_str][norm]['cw']
            table_data.append([
                f"{norm.upper() if norm == 'linf' else 'L2'}", 
                f"{ce_median:.4f}", 
                f"{cw_median:.4f}"
            ])
        
        headers = ["Norm", "Cross-Entropy (CE)", "Carlini-Wagner (CW)"]
        print(f"\n{targeted_str.capitalize()} Attacks (Median Epsilon*):")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
    # 5. Generate ASR vs. Epsilon Plots (Deliverable 2)
    print("\n--- Generating ASR vs. Epsilon Plots ---")
    for (targeted, norm), data in plot_data.items():
        ce_grid, ce_asr = data['ce']
        cw_grid, cw_asr = data['cw']
        
        # Ensure grids are aligned for plotting
        combined_grid = sorted(list(set(ce_grid) | set(cw_grid)))
        ce_asr_interp = np.interp(combined_grid, ce_grid, ce_asr)
        cw_asr_interp = np.interp(combined_grid, cw_grid, cw_asr)

        target_str = "Targeted" if targeted else "Untargeted"
        norm_str = "L-inf" if norm == "linf" else "L2"
        title = f"Success Rate vs. Epsilon ({target_str}, {norm_str})"
        filename = os.path.join(OUTPUT_DIR, f"asr_{target_str.lower()}_{norm}.png")
        
        plot_asr_curves(combined_grid, ce_asr_interp, cw_asr_interp, title, filename)

    # 6. Generate Example Image (Deliverable 3)
    print("\n--- Generating Example Attacked Image ---")
    if first_successful_attack:
        info = first_successful_attack
        
        # Rerun inference to get confidences
        
        # Move images back to device for inference
        clean_img_tensor = info['clean_img'].unsqueeze(0).to(DEVICE)
        adv_img_tensor = info['adv_img'].unsqueeze(0).to(DEVICE)
        
        clean_label_str = ""
        adv_label_str = ""

        with torch.no_grad():
            # Clean image
            clean_logits = model(clean_img_tensor)
            clean_probs = F.softmax(clean_logits, dim=1)
            clean_conf, clean_idx = clean_probs.max(dim=1)
            clean_label_name = labels_map.get(clean_idx.item(), clean_idx.item())
            clean_label_str = f"{clean_label_name} ({clean_conf.item()*100:.1f}%)"

            # Adversarial image
            adv_logits = model(adv_img_tensor)
            adv_probs = F.softmax(adv_logits, dim=1)
            adv_conf, adv_idx = adv_probs.max(dim=1)
            adv_label_name = labels_map.get(adv_idx.item(), adv_idx.item())
            adv_label_str = f"{adv_label_name} ({adv_conf.item()*100:.1f}%)"
        
        
        eps_str = f"{info['eps']:.4f}"
        norm_str = info['norm'].upper()
        loss_str = info['loss_fn'].upper()
        target_str = "Targeted" if info['targeted'] else "Untargeted"
        
        title = f"Example Attack ({target_str})\nNorm: {norm_str}, Loss: {loss_str}, Epsilon* = {eps_str}"
        filename = os.path.join(OUTPUT_DIR, "example_attack.png")
        
        plot_example_image(
            info['clean_img'], info['adv_img'], 
            clean_label_str, adv_label_str,
            title, filename
        )
    else:
        print("Could not generate example image as no attacks were successful.")

    print(f"\nAll experiments complete. Results are saved in '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()