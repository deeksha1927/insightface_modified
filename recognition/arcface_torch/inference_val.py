import argparse
import os

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
from utils.utils_callbacks import CallBackVerification_test
from utils.utils_logging import init_logging
from backbones import get_model


@torch.no_grad()
def inference(network, mg):
    #model_name = mg
    model_name = 'vit_l_dp005_mask_005_exp_0'
    dir_path = 'results'
    sys.path.insert(0, f'{dir_path}/{model_name}')
    model_path = join(dir_path, model_name, 'model.pt') # path of the model
    net= get_model('vit_l_dp005_mask_005', dropout=0.0, fp16=True, num_features=512).cuda()
    #dict_checkpoint = torch.load(model_path)
    net.load_state_dict(torch.load(model_path))
    net.eval().cuda()
    init_logging(0, '/store01/flynn/darun/tmp_ear/')
    # init_logging(0, '/home/kztrk/projects/tmp')

    callback_verification = CallBackVerification_test(
        val_targets=['awe_pairs_176_96'], rec_prefix='/afs/crc.nd.edu/user/d/darun/insightface/recognition/arcface_torch/kagan/deeksha/',
        summary_writer=None, wandb_logger=None
    )
    best_acc = callback_verification(13, net)
    print(model_name, best_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('-m', type=str, default='casia_20')
    args = parser.parse_args()
    inference(args.network, args.m)
