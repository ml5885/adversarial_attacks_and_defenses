"""
This module implements the Projected Gradient Descent (PGD) attack.

References:
    - Nicholas Carlini and David Wagner, "Towards Evaluating the Robustness of Neural Networks," 2017.
    - The original CW attack implementation: https://github.com/carlini/nn_robust_attacks
    - The CleverHans PyTorch implementation: https://github.com/cleverhans-lab/cleverhans
    - The MNIST Challenge implementation: https://github.com/madrylab/mnist_challenge
"""

import torch

from .loss import compute_loss


def _pgd_gradient_step(adv_images, grad, step_size, norm, direction):
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

    return adv_images + direction * step


def _project_delta(delta, epsilon, norm):
    if norm == "linf":
        return torch.clamp(delta, -epsilon, epsilon)
    elif norm == "l2":
        d = delta.view(delta.size(0), -1)
        dnorm = d.norm(p=2, dim=1).view(-1, 1, 1, 1)
        factor = torch.min(torch.ones_like(dnorm), epsilon / (dnorm + 1e-12))
        return delta * factor
    else:
        raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")


def _initialize_random_perturbation(images, epsilon, norm):
    device = images.device
    if norm == "linf":
        return torch.empty_like(images, device=device).uniform_(-epsilon, epsilon)
    elif norm == "l2":
        d = torch.randn_like(images, device=device)
        dflat = d.view(d.size(0), -1)
        dnorm = dflat.norm(p=2, dim=1).view(-1, 1, 1, 1)
        d = d / (dnorm + 1e-12)
        mags = torch.rand(images.size(0), 1, 1, 1, device=device) * epsilon
        return d * mags
    else:
        raise ValueError("Invalid norm type. Choose 'linf' or 'l2'.")


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

    # Random start
    delta = _initialize_random_perturbation(original_images, epsilon, norm)
    adv_images = torch.clamp(original_images + delta, 0.0, 1.0).detach()

    direction = 1.0 if (loss_fn == "ce" and not targeted) else -1.0

    for _ in range(num_steps):
        adv_images.requires_grad = True

        # Forward pass
        logits = model(adv_images)
        
        # Compute loss
        model.zero_grad(set_to_none=True)
        loss = compute_loss(logits, labels, loss_fn, targeted, target_labels, kappa)

        # Backprop to get gradient w.r.t. input
        loss.backward()
        grad = adv_images.grad.data

        with torch.no_grad():
            # Gradient step
            adv_images = _pgd_gradient_step(adv_images, grad, step_size, norm, direction)

            # Project and clip
            delta = _project_delta(adv_images - original_images, epsilon, norm)
            adv_images = torch.clamp(original_images + delta, 0.0, 1.0).detach()

    return adv_images
