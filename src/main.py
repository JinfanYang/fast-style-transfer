from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn

from torchvision import models

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image

from transformer_net import TransformerNet
from utils import gram_matrix, tensor_normalizer
from Vgg16 import Vgg16

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Training settings
parser = argparse.ArgumentParser(description='Nerual Style Transformer')
# train data
parser.add_argument('--data', type=str, default='../data', metavar='D',
                    help='folder where data is located.')
# epoch
parser.add_argument('--epochs', type=int, default=2, metavar='N',
                    help='number of epochs to train (default: 2)')
# batch size
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 4)')
# style image
parser.add_argument('--style-image', type=str, default='../style_image/picasso.jpg',
                    help='the path of style image.')
# image size
parser.add_argument('--image-size', type=int, default=256,
                    help='size of training images, default is 256 X 256')
# style image size
parser.add_argument('--style-size', type=int, default=None,
                    help='size of style-image, default is the original size of style image')
# learning rate
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 1e-3)')
# content weight
parser.add_argument('--content-weight', type=float, default=1,
                    help='weight for content-loss, default is 1')
# style weight
parser.add_argument('--style-weight', type=float, default=1e5,
                    help='weight for style-loss, default is 1e5')
# seed
parser.add_argument('--seed', type=int, default=1080, metavar='S',
                    help='random seed (default: 1080)')
# log interval
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--regularization', type=float, default=1e-7,
                    help='weight for regularization')
# GPU
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disables CUDA training')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    kwargs = {'num_workers': 4, 'pin_memory': True}
else:
    torch.set_default_tensor_type('torch.FloatTensor')
    kwargs = {}


# Data Loading
data_transform = transforms.Compose([transforms.Scale(args.image_size),
                                     transforms.CenterCrop(args.image_size),
                                     transforms.ToTensor(),
                                     tensor_normalizer()])

print('=====> Load train images')
train_dataset = datasets.ImageFolder(args.data, data_transform)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
print('Number of train images:', len(train_dataset))

vgg_model = models.vgg16(pretrained=True)
if args.cuda:
    vgg_model.cuda()

vgg = Vgg16(vgg_model)
vgg.eval()
del vgg_model

print('=====> Load style image')
print('Style image:', args.style_image)
style_img = Image.open(args.style_image).convert('RGB')
style_img_tensor = transforms.Compose([transforms.ToTensor(),
                                       tensor_normalizer()])(style_img).unsqueeze(0)

if args.cuda:
    style_img_tensor = style_img_tensor.cuda()

style_loss_features = vgg(Variable(style_img_tensor, volatile=True))
gram_style = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features]

transformer = TransformerNet()
loss = nn.MSELoss()

if args.cuda:
    transformer.cuda()

optimizer = optim.Adam(transformer.parameters(), args.lr)

print('=====> Start to train')
for epoch in range(2):
    transformer.train()
    agg_content_loss = 0.
    agg_style_loss = 0.
    agg_reg_loss = 0.
    count = 0

    for batch_id, (x, _) in enumerate(train_loader):
        n_batch = len(x)
        count += n_batch
        optimizer.zero_grad()
        x = Variable(x)
        if args.cuda:
            x = x.cuda()

        y = transformer(x)
        xc = Variable(x.data, volatile=True)

        features_y = vgg(y)
        features_xc = vgg(xc)

        f_xc_c = Variable(features_xc[1].data, requires_grad=False)

        content_loss = args.content_weight * loss(features_y[1], f_xc_c)

        reg_loss = args.regularization * (
            torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
            torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

        style_loss = 0.
        for m in range(len(features_y)):
            gram_s = gram_style[m]
            gram_y = gram_matrix(features_y[m])
            style_loss += args.style_weight * loss(gram_y, gram_s.expand_as(gram_y))

        total_loss = content_loss + style_loss + reg_loss
        total_loss.backward()
        optimizer.step()

        agg_content_loss += content_loss.data[0]
        agg_style_loss += style_loss.data[0]
        agg_reg_loss += reg_loss.data[0]

        if (batch_id + 1) % args.log_interval == 0:
            mesg = "[{}/{}] content: {:.6f}  style: {:.6f}  reg: {:.6f}  total: {:.6f}".format(
                count, len(train_dataset),
                agg_content_loss / count,
                agg_style_loss / count,
                agg_reg_loss / count,
                (agg_content_loss + agg_style_loss + agg_reg_loss) / count
            )
            print(mesg)


# save model
transformer.eval()
if torch.cuda.is_available():
    transformer.cpu()

model_file = 'model_' + str(epoch) + '.pth'
torch.save(transformer.state_dict(), model_file)
print('\nSaved model to ' + model_file + '.')
