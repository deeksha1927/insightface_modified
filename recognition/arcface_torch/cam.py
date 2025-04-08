import argparse
import os
import torchvision.transforms.functional
from PIL import Image
import cv2
import numpy as np
import torch
from os.path import join
import sys
from torchvision import transforms
from backbones import get_model

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, class_idx):
        weights = torch.mean(self.gradients, dim=[2, 3])
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1)
        cam = torch.nn.functional.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam

def apply_cam(image, model, target_layer, class_idx):
    grad_cam = GradCAM(model, target_layer)
    output = model(image.unsqueeze(0).cuda())
    model.zero_grad()
    output[:, class_idx].backward()
    cam = grad_cam.generate_cam(class_idx).squeeze().cpu().numpy()
    
    cam = cv2.resize(cam, (112, 112))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    image = np.array(transforms.ToPILImage()(image))
    heatmap = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return heatmap

@torch.no_grad()
def inference(network, patch_size, stride, model_name, image_path, destination):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/'
    model_path = join(dir_path, model_name, 'model.pt')
    
    net = get_model(network, patch_size=int(patch_size), stride=int(stride), dropout=0.0, fp16=True, num_features=512).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval().cuda()
    
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).cuda()
    output = net(image.unsqueeze(0))
    class_idx = torch.argmax(output, dim=1).item()
    cam_image = apply_cam(image, net, list(net.children())[-2], class_idx)
    
    os.makedirs(destination, exist_ok=True)
    cv2.imwrite(join(destination, 'cam_image.png'), cam_image)
    print("CAM image saved at", join(destination, 'cam_image.png'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training with CAM')
    parser.add_argument('--network', type=str, default='vit_t_dp005_mask0', help='backbone network')
    parser.add_argument('--model-name', type=str, required=True, help='model name')
    parser.add_argument('--image-path', type=str, required=True, help='path to the image')
    parser.add_argument('--destination', type=str, required=True, help='destination directory for saving CAM image')
    parser.add_argument('--patch_size', type=str, required=True, help='patch_size')
    parser.add_argument('--stride', type=str, required=True, help='stride')
    
    args = parser.parse_args()
    inference(args.network, args.patch_size, args.stride, args.model_name, args.image_path, args.destination)

