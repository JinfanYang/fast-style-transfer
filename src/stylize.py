from __future__ import print_function
import argparse
import torch
from torchvision import transforms
from torch.autograd import Variable
from transformer_net import TransformerNet
from utils import tensor_normalizer, recover_image
from PIL import Image

parser = argparse.ArgumentParser(description='Stylize a image')
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

img = Image.open(args.content_image).convert('RGB')

transform = transforms.Compose([transforms.ToTensor(),
                                tensor_normalizer()])

img_tensor = transform(img).unsqueeze(0)

if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()

print("=====> Start to transfer")
style_model = TransformerNet()
style_model.load_state_dict(torch.load(args.model))

img_output = style_model(Variable(img_tensor, volatile=True))

output_img = Image.fromarray(recover_image(img_output.data.cpu().numpy())[0])
output_img.save(args.output_image)
print("=====> Save output image to", args.output_image)
