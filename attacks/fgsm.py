import torch

from .loss import compute_loss


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
    """Fast Gradient Sign Method (FGSM) attack."""
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
    loss = compute_loss(logits, labels, loss_fn, targeted, target_labels)
        
    loss.backward()
    grad_sign = images.grad.data.sign()
    if targeted:
        adv_images = images - epsilon * grad_sign  # Move away from target class
    else:
        adv_images = images + epsilon * grad_sign  # Move towards wrong class

    # Clip the adversarial images to be in valid range [0, 1]
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    return adv_images
