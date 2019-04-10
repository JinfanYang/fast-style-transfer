from __future__ import print_function
import argparse
import torch
from torchvision import transforms
from torch.autograd import Variable
from transformer_net import TransformerNet
from utils import tensor_normalizer, recover_image
from PIL import Image

parser = argparse.ArgumentParser(description='Stylize a gif')
# content image
parser.add_argument("--content-image", type=str, default='../content_image/amber.jpg',
                    help="path to content image you want to stylize")
# output image
parser.add_argument("--output-image", type=str, default='../output_image/amber_rain.jpg',
                             help="path for saving the output image")
# model
parser.add_argument("--model", type=str, default='../model/model_rain.pth',
                             help="saved model to be used for stylizing the image")
# GPU
parser.add_argument("--cuda", type=int, default=True,
                             help="set it to 1 for running on GPU, 0 for CPU")

args = parser.parse_args()


print("=====> Start to transfer")

print("=====> Save output gif to", args.output_image)
