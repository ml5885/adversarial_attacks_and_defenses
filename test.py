import argparse
import os
import sys
import torch
import types
import collections
import torchvision

def main():
    if not os.path.isfile("models/resnet18_l2_eps0.ckpt"):
        print(f"Error: file not found: models/resnet18_l2_eps0.ckpt")
        sys.exit(1)

    ckpt = torch.load("models/resnet18_l2_eps0.ckpt", map_location="cpu")

    state_dict = ckpt['model']
    
    # laod the statedict into the resnet18 model architecture
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(state_dict)
    model.eval()
    
if __name__ == "__main__":
    main()