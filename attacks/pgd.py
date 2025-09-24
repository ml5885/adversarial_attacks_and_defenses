"""
This module implements the Projected Gradient Descent (PGD) attack.

References:
    - Nicholas Carlini and David Wagner, "Towards Evaluating the Robustness of Neural Networks," 2017.
    - The official CW attack implementation: https://github.com/carlini/nn_robust_attacks
    - The CleverHans PyTorch implementation: https://github.com/cleverhans-lab/cleverhans
    - PyTorch CW2 implementation by kkew3: https://github.com/kkew3/pytorch-cw2
"""

import torch

from .loss import compute_loss


def _pgd_gradient_step(adv_images, grad, step_size, norm, targeted):
    """Perform a single PGD gradient step."""
    if norm == "linf":
        step = step_size * grad.sign()
    elif norm == "l2":
        grad_flat = grad.view(grad.size(0), -1)
        grad_norm = grad_flat.norm(p=2, dim=1).view(-1, 1, 1, 1)
        norm_grad = grad / (grad_norm + 1e-12)
        step = step_size * norm_grad
    else:
        raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")

    if targeted:
        adv_images = adv_images - step  # minimize targeted loss
    else:
        adv_images = adv_images + step  # maximize loss (move away from true class)
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


def _initialize_random_perturbation(images, epsilon, norm):
    """Initialize random perturbation within the epsilon-ball on the same device as images."""
    device = images.device
    if norm == "linf":
        delta = torch.empty_like(images, device=device).uniform_(-epsilon, epsilon)
    elif norm == "l2":
        delta = torch.randn_like(images, device=device)
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
    kappa=0.0,
    device=None,
):
    """Projected Gradient Descent (PGD) attack.

    Conventions:
      - Untargeted: gradient ascent on the chosen loss.
      - Targeted:   gradient descent on the chosen loss.

    For CW loss:
      - Targeted uses the hinge and we MINIMIZE it.
      - Untargeted uses negative hinge so ASCENT reduces the hinge.
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

    # Random start within epsilon-ball
    delta = _initialize_random_perturbation(original_images, epsilon, norm)
    adv_images = torch.clamp(original_images + delta, 0.0, 1.0).detach()

    # Iterative PGD steps
    for _ in range(num_steps):
        adv_images.requires_grad = True

        logits = model(adv_images)
        model.zero_grad(set_to_none=True)
        if adv_images.grad is not None:
            adv_images.grad.zero_()

        loss = compute_loss(
            logits=logits,
            labels=labels,
            loss_fn=loss_fn,
            targeted=targeted,
            target_labels=target_labels,
            kappa=kappa,
        )

        # Compute gradients
        loss.backward()
        grad = adv_images.grad.data

        # Gradient step
        adv_images = _pgd_gradient_step(adv_images, grad, step_size, norm, targeted)

        # Project the perturbation and clip to valid image range
        delta = _project_delta(adv_images - original_images, epsilon, norm)
        adv_images = torch.clamp(original_images + delta, 0.0, 1.0).detach()

    return adv_images
