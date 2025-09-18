import torch
import torch.nn.functional as F

def _get_cw_loss(logits, labels, targeted=False, target_labels=None, kappa=0.0):
    """
    CW-style margin loss on logits.

    Untargeted (maximize with PGD):
        loss = max_i!=y z_i - z_y
    Targeted (minimize with PGD):
        loss = relu(max_i!=t z_i - z_t + kappa)

    Args:
        logits: [N, C]
        labels: [N]
        targeted: bool
        target_labels: [N] if targeted
        kappa: nonnegative margin parameter (default 0.0)
    """
    if targeted:
        assert target_labels is not None, "Target labels must be provided for targeted attack"
        target_logits = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, target_labels.unsqueeze(1), False)
        other_max = logits[mask].view(logits.size(0), -1).max(1)[0]
        
        loss = torch.clamp(other_max - target_logits + kappa, min=0.0)
        return loss.mean()
    else:
        true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, labels.unsqueeze(1), False)
        other_max = logits[mask].view(logits.size(0), -1).max(1)[0]

        loss = other_max - true_logits
        return loss.mean()


def _compute_loss(logits, labels, loss_fn, targeted, target_labels):
    """Helper function to compute the loss for adversarial attacks.
    
    Args:
        logits: Model logits.
        labels: True labels.
        loss_fn: Loss function type ("ce" or "cw").
        targeted: Whether the attack is targeted.
        target_labels: Target labels for targeted attack.
    
    Returns:
        loss: Computed loss.
    """
    if targeted:
        if loss_fn == "ce":
            loss = F.cross_entropy(logits, target_labels)
        elif loss_fn == "cw":
            loss = _get_cw_loss(logits, labels, targeted=True, target_labels=target_labels)
        else:
            raise ValueError("Invalid loss function. Choose 'ce' or 'cw'.")
    else:
        if loss_fn == "ce":
            loss = F.cross_entropy(logits, labels)
        elif loss_fn == "cw":
            loss = _get_cw_loss(logits, labels, targeted=False)
        else:
            raise ValueError("Invalid loss function. Choose 'ce' or 'cw'.")
    return loss


def _pgd_gradient_step(adv_images, grad, step_size, norm, targeted):
    """Helper function to perform a single PGD gradient step.
    
    Args:
        adv_images: Current adversarial images.
        grad: Gradient of the loss w.r.t. adv_images.
        step_size: Step size for the gradient update.
        norm: Norm type ("linf" or "l2").
        targeted: Whether the attack is targeted.
        
    Returns:
        adv_images: Updated adversarial images after the gradient step.
    """
    if targeted:
        if norm == "linf":
            step = step_size * grad.sign()
            adv_images = adv_images - step
        elif norm == "l2":
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
            norm_grad= grad / (grad_norm + 1e-12)
            step = step_size * norm_grad
            adv_images = adv_images - step
    else:  # Untargeted
        if norm == "linf":
            step = step_size * grad.sign()
            adv_images = adv_images + step
        elif norm == "l2":
            grad_flat = grad.view(grad.size(0), -1)
            grad_norm = grad_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
            norm_grad= grad / (grad_norm + 1e-12)
            step = step_size * norm_grad
            adv_images = adv_images + step
    return adv_images


def _project_delta(delta, epsilon, norm):
    """Helper function to project the perturbation delta back into the epsilon-ball."""
    if norm == "linf":
        delta = torch.clamp(delta, -epsilon, epsilon)
    elif norm == "l2":
        # Project onto l2 ball
        delta_flat = delta.view(delta.size(0), -1)
        delta_norm = delta_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
        
        # Create a factor to scale down deltas that are outside the l2 ball
        factor = torch.min(torch.ones_like(delta_norm), epsilon / (delta_norm + 1e-12))
        delta = delta * factor
    else:
        raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")
    return delta


def _initialize_random_perturbation(images, epsilon, norm, device):
    """Helper function to initialize random perturbation within the epsilon-ball."""
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


def fgsm(
    model,
    images,
    labels,
    epsilon,
    loss_fn="ce",
    targeted=False,
    target_labels=None,
    device=None,
):
    """Implementation of the Fast Gradient Sign Method (FGSM) attack. 
    This is the method described by Goodfellow et al. in 2014 
    (https://arxiv.org/abs/1412.6572).

    Args:
        model: PyTorch model.
        images: Input images (tensor).
        labels: True labels (tensor).
        epsilon: Perturbation budget (float).
        loss_fn: Loss function ("ce" or "cw").
        targeted: Whether to perform a targeted attack.
        target_labels: Target labels for targeted attack.
        device: Device to run attack on.

    Returns:
        adv_images: Adversarial images (tensor).
    """
    if device is None:
        device = images.device
    
    images = images.to(device).clone().detach()
    labels = labels.to(device).clone().detach()
    if targeted:
        assert target_labels is not None, "Target labels must be provided for targeted attack"
        target_labels = target_labels.to(device).clone().detach()
    
    images.requires_grad = True
    model.eval()

    # Forward pass
    logits = model(images)
    
    # Compute loss
    model.zero_grad()
    loss = _compute_loss(logits, labels, loss_fn, targeted, target_labels)
        
    loss.backward()
    grad_sign = images.grad.data.sign()
    if targeted:
        adv_images = images - epsilon * grad_sign  # Move away from target class
    else:
        adv_images = images + epsilon * grad_sign  # Move towards wrong class

    # Clip the adversarial images to be in valid range [0, 1]
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    return adv_images


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
    """Implementation of the Projected Gradient Descent (PGD) attack.
    This is the iterative attack described by Madry et al. (2017).
    (https://arxiv.org/abs/1706.06083).

    Args:
        model: PyTorch model.
        images: Input images (tensor).
        labels: True labels (tensor).
        epsilon: Perturbation budget (float).
        norm: Norm type ("linf" or "l2").
        loss_fn: Loss function ("ce" or "cw").
        targeted: Whether to perform a targeted attack.
        target_labels: Target labels for targeted attack.
        num_steps: Number of PGD steps.
        step_size: Step size for each PGD iteration.
        device: Device to run attack on.

    Returns:
        adv_images: Adversarial images (tensor).
    """
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
        loss = _compute_loss(logits, labels, loss_fn, targeted, target_labels)

        # Compute gradients
        loss.backward()
        grad = adv_images.grad.data

        # Gradient step
        adv_images = _pgd_gradient_step(adv_images, grad, step_size, norm, targeted)

        # Project the perturbation back into the epsilon-ball and valid image range
        delta = _project_delta(adv_images - original_images, epsilon, norm)
        adv_images = torch.clamp(original_images + delta, 0, 1).detach()
    
    return adv_images
