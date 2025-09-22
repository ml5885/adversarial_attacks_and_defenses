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


def compute_loss(logits, labels, loss_fn, targeted, target_labels):
    """Compute CE or CW loss for adversarial attacks.
    
    Args:
        logits: Model logits.
        labels: True labels.
        loss_fn: Loss function type ("ce" or "cw").
        targeted: Whether the attack is targeted.
        target_labels: Target labels for targeted attack.
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
