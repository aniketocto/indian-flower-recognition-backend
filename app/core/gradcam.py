import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path

from app.core.config import OUTPUT_DIR


def generate_gradcam(model, input_tensor, original_image, target_class=None):
    """
    Generate Grad-CAM heatmap and overlay it on the original image.
    
    Returns:
        pred_idx (int)
        confidence (float)
        output_path (Path)
    """          

    gradients = None
    activations = None

    # -----------------------
    # Hooks
    # -----------------------
    gradients = None
    activations = None

    def forward_hook(module, inp, out):
        nonlocal activations
        activations = out.detach()

    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()

    target_layer = model.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)


    # -----------------------
    # Forward Pass
    # -----------------------
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)

    if target_class is None:
        pred_idx = output.argmax(dim=1).item()
    else:
        pred_idx = target_class

    confidence = probs[0, pred_idx].item()

    # -----------------------
    # Backward Pass
    # -----------------------
    model.zero_grad()
    output[0, pred_idx].backward()

    # -----------------------
    # Grad-CAM computation
    # -----------------------
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1)
    cam = F.relu(cam)

    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam[0].cpu().numpy()

    # -----------------------
    # Overlay heatmap on original image
    # -----------------------
    orig = np.array(original_image)
    h, w, _ = orig.shape

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), 
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(0.6 * orig + 0.4 * heatmap)

    # -----------------------
    # Save output
    # -----------------------
    output_path = OUTPUT_DIR / f"gradcam_pred_{pred_idx}.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    #Remove hooks
    fh.remove()
    bh.remove()

    # Convert to relative URL path
    relative_path = f"/outputs/gradcam/{output_path.name}"

    return pred_idx, confidence, relative_path
