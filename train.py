import itertools
from typing import Iterable, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attacks.fgsm import fgsm


def get_data_loaders(batch_size=50, download=True, data_root="data"):
    """Return data loaders for MNIST train and test sets.

    Args:
        batch_size: Number of examples per batch.
        download: Whether to download the dataset if not present.
        data_root: Directory in which to place the MNIST data.

    Returns:
        A tuple (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(data_root, True, transform, download=download)
    test_dataset = datasets.MNIST(data_root, False, transform, download=download)
    
    train_loader = DataLoader(train_dataset, batch_size, True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size, False, drop_last=False)
    
    return train_loader, test_loader


def train_baseline(model, train_loader, device, lr=1e-4, max_steps=100_000):
    """Train a model on clean MNIST data using cross-entropy loss.

    This function iterates over the training data until max_steps
    gradient steps have been taken. The model is updated using the
    Adam optimiser. Once training is complete the model is returned
    in evaluation mode.

    Args:
        model: The neural network to train. Its parameters will be
            updated in place.
        train_loader: DataLoader providing training data.
        device: The device on which to perform computation.
        lr: Learning rate for Adam.
        max_steps: Maximum number of optimisation steps.

    Returns:
        The trained model in evaluation mode.
    """
    model.to(device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    # Use an infinite loop over the DataLoader to avoid thinking in terms of epochs
    data_iter = itertools.cycle(train_loader)
    while step < max_steps:
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimiser.step()
        step += 1
    model.eval()
    return model


def train_fgsm(model, train_loader, device, epsilon_train=0.3, lr=1e-4, max_steps=100_000):
    """Train a model using FGSM adversarial training.

    For each batch we generate adversarial examples using a single
    FGSM step with budget epsilon_train and then optimise the
    model on these adversarial examples using the cross-entropy loss.

    Args:
        model: Neural network to train.
        train_loader: DataLoader of clean training data.
        device: Device for computation.
        epsilon_train: Perturbation magnitude for FGSM training.
        lr: Learning rate for Adam.
        max_steps: Maximum number of optimisation steps.

    Returns:
        The trained model in evaluation mode.
    """
    model.to(device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    data_iter = itertools.cycle(train_loader)
    
    while step < max_steps:
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
    
        # Generate adversarial examples
        adv_images = fgsm(model, images, labels, epsilon_train, device=device)
    
        # Train on adversarial examples
        optimiser.zero_grad(set_to_none=True)
        logits = model(adv_images)
    
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimiser.step()
        step += 1
    
    model.eval()
    return model


def train_trades(model, train_loader, device, epsilon_train=0.3, beta=6.0, lr=1e-4, max_steps=100_000):
    """Train a model using the one-step TRADES adversarial defence.

    At each training step we first approximate the solution to the
    inner maximisation problem by taking a single FGSM step on the
    KL divergence between the model's prediction on the clean input
    and a perturbed input. We then minimise the sum of the
    cross-entropy loss on clean examples and beta times the KL
    divergence between the model's predictions on clean and
    adversarial examples.

    Args:
        model: Neural network to train.
        train_loader: DataLoader of clean training data.
        device: Device for computation.
        epsilon_train: Perturbation magnitude used to generate
            adversarial examples within TRADES.
        beta: Weight on the KL divergence term in the loss.
        lr: Learning rate for Adam.
        max_steps: Maximum number of optimisation steps.

    Returns:
        The trained model in evaluation mode.
    """
    model.to(device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    step = 0
    data_iter = itertools.cycle(train_loader)
    while step < max_steps:
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        # Create adversarial example using one FGSM step on the KL loss
        x_nat = images.clone().detach()
        x_adv = x_nat.clone().detach().requires_grad_(True)
        # Compute logits for clean and adversarial inputs
        logits_nat = model(x_nat)
        logits_adv = model(x_adv)
        # KL divergence between f(x) and f(x_adv)
        kl = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_nat.detach(), dim=1),
            reduction="batchmean",
        )
        model.zero_grad(set_to_none=True)
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        kl.backward()
        # Generate adversarial perturbation
        grad_sign = x_adv.grad.data.sign()
        x_adv = x_adv + epsilon_train * grad_sign
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

        # Compute losses for TRADES
        optimiser.zero_grad(set_to_none=True)
        logits_nat_final = model(x_nat)
        logits_adv_final = model(x_adv)
        ce_loss = F.cross_entropy(logits_nat_final, labels)
        kl_loss = F.kl_div(
            F.log_softmax(logits_adv_final, dim=1),
            F.softmax(logits_nat_final.detach(), dim=1),
            reduction="batchmean",
        )
        loss = ce_loss + beta * kl_loss
        loss.backward()
        optimiser.step()
        step += 1
    model.eval()
    return model

def evaluate_fgsm(model, test_loader, device, epsilons):
    """Evaluate a model's robustness to FGSM attacks on the test set.

    For each epsilon in epsilons, the function computes the
    classification accuracy on adversarial examples generated via a
    single untargeted FGSM step. When epsilon is zero the
    accuracy on the clean test set is reported.

    Args:
        model: Trained neural network.
        test_loader: DataLoader providing test data.
        device: Device on which computation should be carried out.
        epsilons: A collection of perturbation magnitudes to test.

    Returns:
        A mapping from epsilon to accuracy (fraction of correctly
        classified examples).
    """
    model.to(device)
    model.eval()
    results = []
    for eps in epsilons:
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if eps > 0:
                adv_images = fgsm(model, images, labels, eps, device=device)
                outputs = model(adv_images)
            else:
                outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / max(total, 1)
        results[eps] = accuracy
    return results