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
    """Perform a single PGD gradient step.

    Args:
        adv_images: Current adversarial images tensor.
        grad: Gradient of the loss with respect to the adversarial images.
        step_size: Step size for the gradient update.
        norm: Either "linf" or "l2" specifying the norm used.
        targeted: Boolean flag indicating targeted or untargeted attack.

    Returns:
        Updated adversarial images after one step.
    """
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
    """Project the perturbation delta back into the epsilon-ball.

    Args:
        delta: Perturbation tensor (adv_images - original_images).
        epsilon: Maximum allowed perturbation norm.
        norm: Either "linf" or "l2" specifying the norm used.

    Returns:
        Projected perturbation tensor.
    """
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
    """Initialize a random perturbation within the epsilon-ball.

    Args:
        images: Original images tensor.
        epsilon: Maximum allowed perturbation norm.
        norm: Either "linf" or "l2" specifying the norm used.

    Returns:
        A random perturbation tensor of the same shape as images.
    """
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


def _pgd_single(model,
                image,
                label,
                epsilon,
                norm,
                loss_fn,
                targeted=False,
                target_label=None,
                num_steps=40,
                step_size=None,
                kappa=0.0,
                device=None):
    """Run the standard PGD attack on a single image with a fixed kappa.

    This helper is used internally to perform binary search over kappa.
    It contains the core PGD logic without any recursion on binary search.

    Args:
        model: The neural network model to attack.
        image: Tensor of shape [1, C, H, W] representing a single image.
        label: Tensor of shape [1] representing the true label.
        epsilon: Maximum allowed perturbation norm.
        norm: Either "linf" or "l2" specifying the norm used.
        loss_fn: Loss function name: "ce" or "cw".
        targeted: Boolean flag indicating targeted or untargeted attack.
        target_label: Tensor of shape [1] with target label if targeted.
        num_steps: Number of PGD iterations.
        step_size: Step size for gradient updates (defaults to epsilon/4).
        kappa: Confidence margin for CW loss.
        device: Device to perform computation on.

    Returns:
        A tensor of shape [1, C, H, W] representing the adversarial image.
    """
    if device is None:
        device = image.device

    # Determine step size if not provided
    if step_size is None:
        step_size = epsilon / 4.0

    # Clone inputs
    original_image = image.to(device).clone().detach()
    label = label.to(device).clone().detach()
    if targeted:
        assert target_label is not None, "Target labels must be provided for targeted attack"
        target_label = target_label.to(device).clone().detach()

    model.eval()

    # Random initialization
    delta = _initialize_random_perturbation(original_image, epsilon, norm)
    adv_image = torch.clamp(original_image + delta, 0.0, 1.0).detach()

    # Iterative PGD updates
    for _ in range(num_steps):
        adv_image.requires_grad = True
        logits = model(adv_image)
        model.zero_grad(set_to_none=True)
        if adv_image.grad is not None:
            adv_image.grad.zero_()
        loss = compute_loss(
            logits=logits,
            labels=label,
            loss_fn=loss_fn,
            targeted=targeted,
            target_labels=target_label,
            kappa=kappa,
        )
        loss.backward()
        grad = adv_image.grad.data
        adv_image = _pgd_gradient_step(adv_image, grad, step_size, norm, targeted)
        delta = _project_delta(adv_image - original_image, epsilon, norm)
        adv_image = torch.clamp(original_image + delta, 0.0, 1.0).detach()
    return adv_image


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
    binary_search_kappa_steps=0,
    kappa_min=0.0,
    kappa_max=10.0,
):
    """Projected Gradient Descent (PGD) attack with optional kappa search.

    This function implements the classic PGD attack for untargeted and
    targeted settings, supporting both cross-entropy and CW margin losses.
    Additionally, when binary_search_kappa_steps is greater than zero
    and loss_fn is "cw", the attack will perform a per-example
    binary search over the kappa parameter.  The goal of this search
    is to identify the smallest confidence margin that still results in
    a successful adversarial example under the current epsilon budget.

    Args:
        model: The neural network model to attack.
        images: Tensor of shape [N, C, H, W] containing input images.
        labels: Tensor of shape [N] containing true labels.
        epsilon: Maximum allowed perturbation norm.
        norm: Norm type, "linf" or "l2".
        loss_fn: Loss function name, either "ce" or "cw".
        targeted: Whether to perform a targeted attack.
        target_labels: Tensor of shape [N] containing target labels if
            targeted is True.  Ignored otherwise.
        num_steps: Number of PGD iterations.
        step_size: Step size for gradient updates.  Defaults to epsilon/4.
        kappa: Base kappa value when not performing binary search.
        device: Device on which to perform computation.
        binary_search_kappa_steps: Number of binary search steps for
            optimizing kappa.  Setting this to zero disables the search.
        kappa_min: Minimum possible kappa during binary search.
        kappa_max: Maximum possible kappa during binary search.

    Returns:
        A tensor of shape [N, C, H, W] containing adversarial images.
    """
    if device is None:
        device = images.device

    # Default step size
    if step_size is None:
        step_size = epsilon / 4.0

    images = images.to(device).clone().detach()
    labels = labels.to(device).clone().detach()
    if targeted:
        assert target_labels is not None, "Target labels must be provided for targeted attack"
        target_labels = target_labels.to(device).clone().detach()

    # If not searching kappa or not using CW loss, fall back to standard PGD on the full batch
    if loss_fn != "cw" or binary_search_kappa_steps <= 0:
        original_images = images.clone().detach()
        model.eval()
        # Random initialization for the entire batch
        delta = _initialize_random_perturbation(images, epsilon, norm)
        adv_images = torch.clamp(original_images + delta, 0.0, 1.0).detach()
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
            loss.backward()
            grad = adv_images.grad.data
            adv_images = _pgd_gradient_step(adv_images, grad, step_size, norm, targeted)
            delta = _project_delta(adv_images - original_images, epsilon, norm)
            adv_images = torch.clamp(original_images + delta, 0.0, 1.0).detach()
        return adv_images

    # Otherwise, perform per-example binary search over kappa
    batch_size = images.size(0)
    adv_list = []
    for i in range(batch_size):
        img = images[i : i + 1]
        lbl = labels[i : i + 1]
        tgt_lbl = target_labels[i : i + 1] if targeted else None
        low = kappa_min
        high = kappa_max
        best_adv = None
        # Keep track of whether we have found any successful attack
        found = False
        # Initialize last_adv in case search never succeeds
        last_adv = None
        for step in range(binary_search_kappa_steps):
            mid = (low + high) / 2.0
            adv = _pgd_single(
                model,
                img,
                lbl,
                epsilon=epsilon,
                norm=norm,
                loss_fn=loss_fn,
                targeted=targeted,
                target_label=tgt_lbl,
                num_steps=num_steps,
                step_size=step_size,
                kappa=mid,
                device=device,
            )
            # Evaluate success for this single example
            with torch.no_grad():
                pred = model(adv).argmax(dim=1)
            if targeted:
                success = (pred.item() == tgt_lbl.item())
            else:
                success = (pred.item() != lbl.item())
            last_adv = adv  # Save the last attempted adversarial example
            if success:
                found = True
                best_adv = adv
                # Update upper bound on kappa
                high = mid
            else:
                # Update lower bound on kappa
                low = mid
        # If never succeeded, fallback to the last attempt
        if not found:
            best_adv = last_adv if last_adv is not None else img.clone().detach()
        adv_list.append(best_adv.detach())
    # Concatenate the list of adversarial images
    return torch.cat(adv_list, dim=0)
