import torch
import torchvision.models as models

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedModel(torch.nn.Module):
    """Wraps a model to normalize input tensors in the forward pass."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return self.model(x)


def load_resnet18(checkpoint_path, device):
    """Loads ResNet18 model weights from local checkpoint and wraps with normalization."""
    base_model = models.resnet18(weights=None)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle possible checkpoint formats
        state_dict = checkpoint.get("model", checkpoint)

        # Remove prefixes like "module." or "model." 
        # These are artifacts from whatever training setup was used
        # by https://huggingface.co/madrylab/robust-imagenet-models
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "").replace("model.", "")
            cleaned_state_dict[k] = v

        base_model.load_state_dict(cleaned_state_dict, strict=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {e}")

    model = NormalizedModel(base_model).to(device)
    model.eval()
    return model
