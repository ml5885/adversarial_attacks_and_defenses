import torch
import torch.nn.functional as F


def _get_cw_loss(logits, labels, targeted=False, target_labels=None, kappa=0.0):
    """CW-style margin loss on logits with confidence kappa.

    Targeted (we minimize a hinge):
        loss = relu(max_{i!=t} z_i - z_t + kappa)

    Untargeted (we return negative hinge so ASCENT reduces the hinge):
        hinge = relu(z_y - max_{i!=y} z_i + kappa)
        loss  = -hinge

    Args:
        logits: Tensor of shape [N, C], model logits.
        labels: Tensor of shape [N], true labels.
        targeted: bool, whether targeted attack.
        target_labels: Tensor of shape [N] if targeted.
        kappa: float >= 0.0, confidence margin.
    """
    if targeted:
        assert target_labels is not None, "Target labels must be provided for targeted attack"

        # z_t
        target_logits = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)

        # max_{i != t} z_i
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, target_labels.unsqueeze(1), False)
        other_max = logits[mask].view(logits.size(0), -1).max(1)[0]

        # relu(max_other - z_t + kappa)
        loss = torch.clamp(other_max - target_logits + kappa, min=0.0)
        return loss.mean()

    # Untargeted
    true_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)

    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    other_max = logits[mask].view(logits.size(0), -1).max(1)[0]

    # hinge = relu(z_y - max_other + kappa)
    hinge = torch.clamp(true_logits - other_max + kappa, min=0.0)

    # Return negative hinge so gradient ASCENT decreases hinge -> pushes to misclassify
    loss = -hinge
    return loss.mean()


def compute_loss(logits, labels, loss_fn, targeted, target_labels, kappa=0.0):
    """Compute CE or CW loss for adversarial attacks.

    Args:
        logits: [N, C] model logits.
        labels: [N] true labels.
        loss_fn: "ce" or "cw".
        targeted: bool.
        target_labels: [N] if targeted else None.
        kappa: float margin for CW loss (default 0.0).
    """
    if targeted:
        if loss_fn == "ce":
            if target_labels is None:
                raise ValueError("target_labels must be provided for targeted CE.")
            return F.cross_entropy(logits, target_labels)
        elif loss_fn == "cw":
            return _get_cw_loss(
                logits=logits,
                labels=labels,
                targeted=True,
                target_labels=target_labels,
                kappa=kappa,
            )
        else:
            raise ValueError("Invalid loss function. Choose 'ce' or 'cw'.")
    else:
        if loss_fn == "ce":
            return F.cross_entropy(logits, labels)
        elif loss_fn == "cw":
            return _get_cw_loss(
                logits=logits,
                labels=labels,
                targeted=False,
                target_labels=None,
                kappa=kappa,
            )
        else:
            raise ValueError("Invalid loss function. Choose 'ce' or 'cw'.")
