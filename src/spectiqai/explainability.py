"""Explainability utilities for SpectiqAI models using Grad-CAM."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) implementation."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Register forward and backward hooks to capture activations and gradients
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: torch.nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Generate coarse heatmap for a specific class index."""
        # Ensure model is in eval mode
        self.model.eval()
        
        # Enable gradients explicitly to support calling from no_grad environments
        with torch.enable_grad():
            # Clone input and enable gradients to construct the backward graph
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            logits = self.model(input_tensor)
            self.model.zero_grad()

            # Backward pass for the target class logit score
            if class_idx >= logits.shape[1]:
                raise ValueError(f"class_idx {class_idx} is out of bounds for model logits of shape {logits.shape}.")
            
            score = logits[0, class_idx]
            score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients. Check target layer compatibility.")

        activations = self.activations[0]  # Shape: (channels, H, W)
        gradients = self.gradients[0]      # Shape: (channels, H, W)

        # Compute weights: Global Average Pooling of gradients per channel
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # Shape: (channels, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * activations, dim=0)  # Shape: (H, W)

        # Apply ReLU (only keep positive features contributing to the class)
        cam = F.relu(cam)

        # Normalize map to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Find the target activation/convolution layer for Grad-CAM depending on architecture."""
    model_name = model.__class__.__name__

    if model_name == "SimpleKidneyCNN":
        # Target the Conv2d layer of the 4th ConvBlock features[3]
        # features[3] is ConvBlock, features[3].layers is Sequential:
        # index 0: Conv2d, index 1: BatchNorm2d, index 2: ReLU, index 3: MaxPool2d
        # Targeting Conv2d (index 0) avoids inplace ReLU autograd issues.
        return model.features[3].layers[0]
    elif model_name == "ResNet":
        # Target the final block of the last resnet block group
        return model.layer4[-1]
    elif "EfficientNet" in model_name:
        # Target the last layer of the feature extractor
        return model.features[-1]
    else:
        # Fallback to the last Conv2d layer in the network
        conv_layers = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                conv_layers.append(module)
        if conv_layers:
            return conv_layers[-1]
        raise ValueError(f"Could not automatically determine target conv layer for model: {model_name}")


def generate_gradcam_images(
    model: torch.nn.Module,
    image: Image.Image,
    class_idx: int,
    image_size: int = 224,
    alpha: float = 0.5,
) -> Tuple[Image.Image, Image.Image]:
    """
    Generate Grad-CAM heatmap and overlay images for a given PIL Image.

    Args:
        model: The PyTorch classification model.
        image: Original input PIL Image.
        class_idx: The class index to explain (0: Normal, 1: Tumor).
        image_size: Resizing shape for input tensor.
        alpha: Blending weight for overlay image (0.0: only original, 1.0: only heatmap).

    Returns:
        heatmap_img: Styled colormap heatmap PIL Image.
        overlay_img: Original image with heatmap blended.
    """
    target_layer = get_target_layer(model)
    grad_cam = GradCAM(model, target_layer)

    # Resolve correct preprocessing (ImageNet normalization for transfer learning models)
    model_name = model.__class__.__name__
    if model_name != "SimpleKidneyCNN":
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    input_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    try:
        # Generate the coarse CAM map (2D numpy float array in 0.0 - 1.0)
        cam = grad_cam.generate(input_tensor, class_idx)
    finally:
        # Always remove hooks to prevent subsequent memory/deserialization leaks
        grad_cam.remove_hooks()

    # Resize raw heatmap to match original image size
    original_size = image.size  # (width, height)
    heatmap_pil = Image.fromarray((cam * 255).astype(np.uint8))
    heatmap_pil = heatmap_pil.resize(original_size, Image.Resampling.BILINEAR)

    # Apply JET colormap using NumPy vector calculations
    cam_resized = np.array(heatmap_pil) / 255.0
    r = np.clip(np.minimum(4 * cam_resized - 1.5, -4 * cam_resized + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4 * cam_resized - 0.5, -4 * cam_resized + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4 * cam_resized + 0.5, -4 * cam_resized + 2.5), 0.0, 1.0)
    
    heatmap_color_arr = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    heatmap_color_img = Image.fromarray(heatmap_color_arr)

    # Blend original image with color heatmap
    original_rgb = image.convert("RGB")
    overlay_img = Image.blend(original_rgb, heatmap_color_img, alpha)

    return heatmap_color_img, overlay_img
