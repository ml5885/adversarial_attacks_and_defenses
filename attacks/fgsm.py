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
    kappa=0.0,
    device=None,
):
    """Fast Gradient Sign Method (FGSM) attack.

    Conventions:
      - Untargeted: take a step in the direction of +grad (gradient ascent on the loss).
      - Targeted: take a step in the direction of -grad (gradient descent on the loss).

    For CW loss:
      - Targeted uses the standard hinge on logits and we MINIMIZE it.
      - Untargeted uses the negative hinge so that ASCENT reduces the hinge toward 0.
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
    model.zero_grad(set_to_none=True)
    if images.grad is not None:
        images.grad.zero_()
    loss = compute_loss(
        logits=logits,
        labels=labels,
        loss_fn=loss_fn,
        targeted=targeted,
        target_labels=target_labels,
        kappa=kappa,
    )

    # Backprop to get gradient w.r.t. input
    loss.backward()
    grad_sign = images.grad.data.sign()

    if targeted:
        # Move toward the target class (minimize targeted loss)
        adv_images = images - epsilon * grad_sign
    else:
        # Move away from the true class (maximize loss)
        adv_images = images + epsilon * grad_sign

    # Clip to valid range [0, 1]
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

    return adv_images
