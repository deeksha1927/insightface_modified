import os
from os.path import join
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from backbones import get_model

def inference(network, patch_size, stride, model_name, image_path):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/vit_t_dp005_mask0_p28_s28_original/'
    model_path = join(dir_path, model_name, 'model.pt')
    
    # Build model exactly as during training.
    net = get_model(network, patch_size=int(patch_size), stride=int(stride),
                    dropout=0.0, fp16=True, num_features=512).cuda()
    
    net.load_state_dict(torch.load(model_path))
    net.eval().cuda()
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).cuda()
    
    return net, image

def forward_hook(module, input, output):
    global activations
    activations = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    print("Backward hook triggered")  # Debugging message
    gradients = grad_out[0].detach()

if __name__ == '__main__':

    activations = None
    gradients = None

    # Parameters
    network = "vit_t_dp005_mask0"
    patch_size = 28
    stride = 28
    model_name = "vit_t_dp005_mask0_p28_s28_original_4"
    image_path = "/store01/flynn/darun/AWE-Ex_New_images_lr/220_R/10.png"

    # Load model and image
    net, image = inference(network, patch_size, stride, model_name, image_path)

    # Register hooks (adjust the target layer based on your model architecture)
    target_layer = net.blocks[-1]  # Adjust if necessary
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = net(image.unsqueeze(0))
    
    # Enable gradient tracking
    torch.set_grad_enabled(True)
    net.zero_grad()

    # Get predicted class
    pred_class = output.argmax(dim=1).item()
    score = output[0, pred_class]
    
    # Compute gradients
    score.backward()

    if gradients is None:
        raise ValueError("Gradients are None. Check if the backward hook is registered correctly.")

    # Compute Grad-CAM
    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * activations, dim=1).squeeze()
    cam = F.relu(cam)

    # Normalize CAM
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # Load original image for overlay
    img = cv2.imread(image_path)
    img = cv2.resize(img, (112, 112))  # Ensure size matches input

    # Resize CAM to match image
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Generate heatmap and overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.5, heatmap, 0.5, 0)

    # Display and save
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Grad-CAM for class {pred_class}')
    plt.savefig("test_cam.png")

    print("Grad-CAM saved as test_cam.png")


