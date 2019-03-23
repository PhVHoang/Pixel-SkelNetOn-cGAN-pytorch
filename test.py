import torch
import os, util
import network
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
import os

model_path = 'train_img_results/train_img_generator_param.pkl'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='train_img',  help='')
parser.add_argument('--test_subfolder', required=False, default='test',  help='')
parser.add_argument('--ngf', type=int, default=64)
opt = parser.parse_args()

if not os.path.isdir(opt.dataset + '_results/test_results'):
    os.mkdir(opt.dataset + '_results/test_results')

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_loader = util.load_data(opt.dataset, opt.test_subfolder, transform, batch_size=1, shuffle=False)

G = network.Generator(opt.ngf)
G.cuda()
G.load_state_dict(torch.load(opt.dataset + '_results/' + opt.dataset + '_generator_param.pkl'))

# network
n = 0

print('start testing...')
for x_, _ in test_loader:
    x_ = torch.Tensor(x_).cuda()
    res_img = G(x_)
    s = test_loader.dataset.imgs[n][0][::-1]
    s_ind = len(s) - s.find('/')
    e_ind = len(s) - s.find('.')
    ind = test_loader.dataset.imgs[n][0][s_ind:e_ind-1]
    # path = opt.dataset + '_results/test_results/' + ind + '_input.png'
    # plt.imsave(path, (x_[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    path = opt.dataset + '_results/test_results/' + ind + '.png'
    plt.imsave(path, (test_image[0].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)
    path = opt.dataset + '_results/test_results/' + ind + '_target.png'
    plt.imsave(path, (y_[0].numpy().transpose(1, 2, 0) + 1) / 2)

    n += 1
print("%d images have been generated successfully")
