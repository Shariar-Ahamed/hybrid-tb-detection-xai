import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image as PILImage
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image

class GuidedBackpropagation:
    """
    Computes Guided Backpropagation gradients for a PyTorch CNN model.
    Only allows positive gradients and activations to backpropagate.
    """
    def __init__(self, model):
        self.model = model
        # Disable in-place ReLU operations to avoid autograd view issues with backward hooks
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.forward_relu_outputs = []
        self.handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.forward_relu_outputs.append(output)

        def backward_hook(module, grad_in, grad_out):
            if len(self.forward_relu_outputs) > 0:
                corresponding_forward_output = self.forward_relu_outputs.pop()
                gate_g = torch.clamp(corresponding_forward_output, min=0.0)
                gate_b = torch.clamp(grad_out[0], min=0.0)
                # Clone the output to prevent PyTorch's in-place view modification checks
                return ((gate_g * gate_b).clone(),)
            return grad_in

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.handlers.append(module.register_forward_hook(forward_hook))
                self.handlers.append(module.register_full_backward_hook(backward_hook))

    def generate_gradients(self, input_tensor):
        self.forward_relu_outputs = []
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Binary classification target score (Tuberculosis channel logit)
        output_score = output[0][0]
        output_score.backward()
        
        gradients = input_tensor.grad[0].cpu().data.numpy()
        return gradients

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()
        self.handlers = []

def generate_saliency_map(model, input_tensor):
    """
    Generates standard Vanilla Saliency map (gradients of input).
    """
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    model.zero_grad()
    output = model(input_tensor)
    output_score = output[0][0]
    output_score.backward()
    
    saliency, _ = torch.max(torch.abs(input_tensor.grad[0]), dim=0)
    saliency = saliency.cpu().data.numpy()
    return saliency

def generate_xai_visualization(method_name, model, input_tensor, original_image):
    """
    Generates the corresponding XAI visualization map based on the chosen method name.
    
    Args:
        method_name (str): Selected XAI method.
        model (nn.Module): Evaluated PyTorch CNN model.
        input_tensor (torch.Tensor): Preprocessed input image tensor (1, 3, 224, 224).
        original_image (PIL.Image): Original uploaded image to overlay or resize.
        
    Returns:
        PIL.Image: Visualization image object.
    """
    img_np = np.array(original_image.resize((224, 224))).astype(np.float32) / 255.0
    
    if method_name == "Grad-CAM":
        target_layers = [model.features]
        cam = GradCAM(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return PILImage.fromarray(overlay)
        
    elif method_name == "Grad-CAM++":
        target_layers = [model.features]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
        return PILImage.fromarray(overlay)
        
    elif method_name == "Guided Backpropagation":
        gbp = GuidedBackpropagation(model)
        try:
            gradients = gbp.generate_gradients(input_tensor)
            # Transpose from (3, H, W) to (H, W, 3)
            grads = np.transpose(gradients, (1, 2, 0))
            # Standardize & Normalize to 0-255 range
            grads = grads - grads.min()
            grads = grads / (grads.max() + 1e-8)
            grads_img = (grads * 255).astype(np.uint8)
            return PILImage.fromarray(grads_img)
        finally:
            gbp.remove_hooks()
            
    elif method_name == "Saliency Map":
        saliency = generate_saliency_map(model, input_tensor)
        # Normalize
        saliency = saliency - saliency.min()
        saliency = saliency / (saliency.max() + 1e-8)
        # Apply standard medical COLORMAP_HOT
        saliency_heatmap = cv2.applyColorMap((saliency * 255).astype(np.uint8), cv2.COLORMAP_HOT)
        saliency_heatmap = cv2.cvtColor(saliency_heatmap, cv2.COLOR_BGR2RGB)
        # Blend heatmap with original image (65% original, 35% heatmap)
        heatmap_float = saliency_heatmap.astype(np.float32) / 255.0
        overlay = cv2.addWeighted(img_np, 0.65, heatmap_float, 0.35, 0)
        overlay = (overlay * 255).astype(np.uint8)
        return PILImage.fromarray(overlay)
        
    else:
        # Fallback to resized original image if unknown method name
        return original_image.resize((224, 224))
