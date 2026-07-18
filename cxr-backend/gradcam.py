"""
Minimal, self-contained Grad-CAM implementation for a single target conv layer.

Avoids depending on the external `pytorch-grad-cam` package so the backend
has fewer moving parts to install. Works for any CNN classifier.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Runs a forward + backward pass and returns (logits, cam_map).

        cam_map is a single-channel float32 array in [0, 1], resized to the
        input's spatial dimensions (H, W).
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=False)

        # activations/gradients: (1, C, h, w)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return logits.detach(), cam


def overlay_heatmap(base_image_01: np.ndarray, cam_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    base_image_01: HWC float32 RGB image, values in [0, 1]
    cam_map: HxW float32 in [0, 1]
    Returns: HWC uint8 RGB blended image
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    blended = (1 - alpha) * base_image_01 + alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return (blended * 255).astype(np.uint8)
