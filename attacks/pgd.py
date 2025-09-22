import torch

from .loss import compute_loss


def _pgd_gradient_step(adv_images, grad, step_size, norm, targeted):
    """Perform a single PGD gradient step."""
    if targeted:
        if norm == "linf":
            step = step_size * grad.sign()
            adv_images = adv_images - step
        elif norm == "l2":
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
            norm_grad = grad / (grad_norm + 1e-12)
            step = step_size * norm_grad
            adv_images = adv_images - step
        else:
            raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")
    else:  # Untargeted
        if norm == "linf":
            step = step_size * grad.sign()
            adv_images = adv_images + step
        elif norm == "l2":
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
            norm_grad = grad / (grad_norm + 1e-12)
            step = step_size * norm_grad
            adv_images = adv_images + step
        else:
            raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")
    return adv_images


def _project_delta(delta, epsilon, norm):
    """Project the perturbation delta back into the epsilon-ball."""
    if norm == "linf":
        delta = torch.clamp(delta, -epsilon, epsilon)
    elif norm == "l2":
        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = delta_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
        factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-12))
        delta = delta * factor
    else:
        raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")
    return delta


def _initialize_random_perturbation(images, epsilon, norm, device):
    """Initialize random perturbation within the epsilon-ball."""
    if norm == "linf":
        delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    elif norm == "l2":
        delta = torch.randn_like(images)
        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = delta_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
        delta = delta / (delta_norm + 1e-12)
        rand_mags = torch.rand(images.size(0), 1, 1, 1, device=device) * epsilon
        delta = delta * rand_mags
    else:
        raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")
    return delta


def pgd(
    model,
    images,
    labels,
    epsilon,
    norm="linf",
    loss_fn="ce",
    targeted=False,
    target_labels=None,
    num_steps=40,
    step_size=None,
    device=None,
):
    """Projected Gradient Descent (PGD) attack."""
    if device is None:
        device = images.device
    
    if step_size is None:
        step_size = epsilon / 4.0
    
    original_images = images.to(device).clone().detach()
    labels = labels.to(device).clone().detach()
    if targeted:
        assert target_labels is not None, "Target labels must be provided for targeted attack"
        target_labels = target_labels.to(device).clone().detach()

    model.eval()
    
    adv_images = original_images.clone().detach()
    delta = _initialize_random_perturbation(images, epsilon, norm, device)

    # Initialize adversarial images
    adv_images = torch.clamp(adv_images + delta, 0, 1).detach()
    
    # Iterative PGD steps
    for _ in range(num_steps):
        adv_images.requires_grad = True
        logits = model(adv_images)
        
        model.zero_grad()
        loss = compute_loss(logits, labels, loss_fn, targeted, target_labels)

        # Compute gradients
        loss.backward()
        grad = adv_images.grad.data

        # Gradient step
        adv_images = _pgd_gradient_step(adv_images, grad, step_size, norm, targeted)

        # Project the perturbation back into the epsilon-ball and valid image range
        delta = _project_delta(adv_images - original_images, epsilon, norm)
        adv_images = torch.clamp(original_images + delta, 0, 1).detach()
    
    return adv_images
