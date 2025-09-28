import torch
import torch.nn.functional as F


def cw_margin_loss(logits, labels, targeted=False, target_labels=None, kappa=50.0):
    """Carlini-Wagner margin loss, for both targeted and untargeted attacks.

    The loss function is defined as:
    
        $$f(x) = \max(\max_{i \neq t} z_i - z_t + \kappa, 0)$$ for targeted attacks

        $$f(x) = \max(z_y - \max_{i \neq y} z_i + \kappa, 0)$$ for untargeted attacks
    
    where z are the logits, y is the true label, t is the target label,
    and kappa is a nonnegative margin.
    
    In the targeted case, the loss is minimized when the target class logit is greater
    than all other logits by at least kappa. 
    
    In the untargeted case, the loss is minimized when the true class logit is less
    than the maximum logit of any other class by at least kappa.
    
    Source: Carlini and Wagner, "Towards Evaluating the Robustness of Neural Networks," 2017.
    https://arxiv.org/pdf/1608.04644
    
    Args:
        logits: [N, C] tensor of model logits.
        labels: [N] true labels.
        targeted: whether to use the targeted variant.
        target_labels: [N] target labels if targeted=True.
        kappa: nonnegative margin.
    """
    num_classes = logits.size(1)

    if targeted:
        assert target_labels is not None, "Target labels must be provided for targeted CW"

        t = target_labels
        t_mask = F.one_hot(t, num_classes=num_classes).bool()

        logits_other = logits.masked_fill(t_mask, float('-inf'))
        other = logits_other.max(dim=1)[0]

        z_t = logits.gather(1, t.unsqueeze(1)).squeeze(1)

        loss = F.relu(other - z_t + float(kappa))
        return loss.mean()

    # Untargeted
    y = labels
    y_mask = F.one_hot(y, num_classes=num_classes).bool()

    logits_other = logits.masked_fill(y_mask, float('-inf'))
    other = logits_other.max(dim=1)[0]

    z_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)

    loss = F.relu(z_y - other + float(kappa))
    return loss.mean()


def ce_loss(logits, labels, targeted=False, target_labels=None):
    """Cross-entropy loss, for both targeted and untargeted attacks.

    Args:
        logits: [N, C] tensor of model logits.
        labels: [N] true labels.
        targeted: whether to use the targeted variant.
        target_labels: [N] target labels if targeted=True.
    """
    if targeted:
        assert target_labels is not None, "Target labels must be provided for targeted CE"
        return F.cross_entropy(logits, target_labels)
    return F.cross_entropy(logits, labels)


def compute_loss(logits, labels, loss_fn, targeted, target_labels, kappa=50.0):
    """
    Compute loss for adversarial attacks.

    - 'ce': cross-entropy. If targeted=True, it uses cross-entropy on target_labels.
    - 'cw': Carlini-Wagner margin loss

    Args:
        logits: [N, C]
        labels: [N]
        loss_fn: "ce" or "cw"
        targeted: bool
        target_labels: [N] if targeted else None
        kappa: margin for CW (default 50.0)
    """
    if loss_fn == "ce":
        return ce_loss(logits, labels, targeted, target_labels)

    if loss_fn == "cw":
        return cw_margin_loss(logits, labels, targeted, target_labels, kappa)

    raise ValueError("Invalid loss function. Choose 'ce' or 'cw'.")
