import copy
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision
import glob
import torch.nn.functional as F
from torchvision import transforms


def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor(), normalize])
preprocess_transform = get_preprocess_transform()
model_name = 'name'
model = torch.load(f'{model_name}.pt')
# Pick up layers for visualization
target_layers = [model.layer4[-1]]

rgb_img = Image.open(file).convert('RGB')
input_tensor = preprocess_transform(rgb_img).unsqueeze(0)
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
I = Image.fromarray(visualization, 'RGB')

