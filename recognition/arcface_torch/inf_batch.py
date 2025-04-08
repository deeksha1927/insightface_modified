import argparse
import os

import torchvision.transforms.functional
from PIL import Image
import cv2
import numpy as np
import torch
from os.path import join
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.io import read_image
from numpy.linalg import norm
from backbones import get_model

def absoluteFilePaths(directory):
    lst = []
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.lower().endswith('.jpg') or f.lower().endswith('.png'):
                lst.append(os.path.abspath(os.path.join(dirpath, f)))
    return sorted(lst)

class ImageList(ImageFolder):
    def __init__(self, source, transform):
        self.samples = absoluteFilePaths(source)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = Image.open(self.samples[index]).convert('RGB')
        return self.transform(img)

class TestDataLoader(DataLoader):
    def __init__(
        self, batch_size, source, transform=None
    ):
        self._dataset = ImageList(source, transform)

        super(TestDataLoader, self).__init__(
            self._dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=3,
        )

@torch.no_grad()
def inference(network, patch_size,stride, model_name, dataset, batch_size, source, destination):
    dir_path = '/afs/crc.nd.edu/user/d/darun/if-copy2/recognition/arcface_torch/results/'
    print(model_name)

    
    model_path = join(dir_path, model_name, 'model.pt') # path of the model
    print(model_path)
    print(f"network: {network}, patch_size: {patch_size}, stride: {stride}")

    #net = get_model(network, dropout=0.0, fp16=True, num_features=512).cuda()
    net = get_model(network, patch_size=int(patch_size),stride=int(stride), dropout=0.0, fp16=True, num_features=512).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval().cuda()
    
    destination = join(destination, model_name, dataset) #
    os.makedirs(destination, exist_ok=True)

    source = join(source, dataset)

    os.makedirs(destination, exist_ok=True)

    print(model_name)
    print(dataset)
    
    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    loader = TestDataLoader(
        batch_size=batch_size,
        source=source,
        transform=transform,
    )
    print('len', len(loader.dataset))
    image_paths = loader.dataset.samples
    feats = np.zeros((len(loader.dataset), 512), dtype=np.float32)
    idx = 0
    for imgs in tqdm(loader):
        feat = net(imgs.cuda()).cpu().detach().numpy().squeeze()
        l = len(feat)
        feats[idx:idx+l] = feat
        idx += l

    with open(join(destination, f'{dataset}.txt'), 'w') as f:
        for i in image_paths:
            f.write(f'{i}\n')
    print('saved lst')
    np.save(join(destination, f'{dataset}.npy'), feats / norm(feats, axis=1)[:, np.newaxis])
    print('saved feat')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='vit_t_dp005_mask0', help='backbone network')
    parser.add_argument('--model-name', type=str, required=True, help='model name')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--source', type=str, required=True, help='source directory of images')
    parser.add_argument('--destination', type=str, required=True, help='destination directory for saving features')
    parser.add_argument('--patch_size', type=str, required=True, help='patch_size')
    parser.add_argument('--stride', type=str, required=True, help='stride')
    
    args = parser.parse_args()
    inference(args.network, args.patch_size,args.stride, args.model_name, args.dataset, args.batch_size, args.source, args.destination)
    #inference(args.network, args.model_name, args.dataset, args.batch_size, args.source, args.destination)

