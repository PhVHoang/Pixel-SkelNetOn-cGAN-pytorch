import os, time, pickle, network, util
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='train_img',  help='')
parser.add_argument('--train_subfolder', required=False, default='combine',  help='')
parser.add_argument('--test_subfolder', required=False, default='test',  help='')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--test_batch_size', type=int, default=5, help='test batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
# parser.add_argument('--input_size', type=int, default=256, help='input size')
# parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
# parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
# parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True or False')
parser.add_argument('--train_epoch', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--L1_lambda', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
parser.add_argument('--save_root', required=False, default='results', help='results save path')
parser.add_argument('--inverse_order', type=bool, default=True, help='0: [input, target], 1 - [target, input]')
opt = parser.parse_args()
print(opt)

root = opt.dataset + '_' + opt.save_root + '/'
model = opt.dataset + '_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'Fixed_results'):
    os.mkdir(root + 'Fixed_results')

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


train_loader = util.load_data(opt.dataset, opt.train_subfolder, transform, opt.batch_size, shuffle=True)
test_loader = util.load_data(opt.dataset, opt.test_subfolder, transform, opt.batch_size, shuffle=True)
test = test_loader.__iter__().__next__()[0]
img_size = test.size()[2]
print('image size = {}'.format(img_size))


if opt.inverse_order:
    fixed_y_ = test[:,:,:,0:img_size]
    fixed_x_ = test[:,:,:,img_size:]
else:
    fixed_x_ = test[:,:,:,0:img_size]
    fixed_y_ = test[:,:,:,img_size:]

G = network.Generator(opt.ngf)
D = network.discriminator(opt.ndf)

G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
G.cuda()
D.cuda()
G.train()
D.train()

BCE_Loss = nn.BCELoss().cuda()
L1_Loss = nn.L1Loss().cuda()

G_optimizer = optim.Adam(G.parameters(), lr=opt.lrG, betas=(opt.beta1, opt.beta2))
D_optimizer = optim.Adam(D.parameters(), lr=opt.lrD, betas=(opt.beta1, opt.beta2))

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

print('start training...')
start_time = time.time()
for epoch in range(opt.train_epoch):
    D_losses = []
    G_losses = []
    epoch_start_time = time.time()
    num_iter = 0
    print('training epoch {}'.format(epoch+1))

    for x_, _ in train_loader:
        D.zero_grad()
        if opt.inverse_order:
            y_ = x_[:, :, :, 0:img_size]
            x_ = x_[:, :, :, img_size:]
        else:
            y_ = x_[:, :, :, img_size:]
            x_ = x_[:, :, :, 0:img_size]

        x_, y_ = torch.Tensor(x_).cuda(), torch.Tensor(y_).cuda()
        print('x_.shape = {}\ny_.shape={}'.format(x_.shape, y_.shape))
        D_result = D(x_, y_).squeeze()
        D_real_loss = BCE_Loss(D_result, torch.Tensor(torch.ones(D_result.size())).cuda())

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()
        D_fake_loss = BCE_Loss(D_result, torch.Tensor(torch.zeros(D_result.size())).cuda())

        D_train_loss = (D_real_loss + D_fake_loss) * 0.5
        print("Discriminator loss: ", D_train_loss.data)
        D_train_loss.backward()
        D_optimizer.step()

        #train_hist['D_losses'].append(D_train_loss.data[0])

        D_losses.append(D_train_loss.data)

        # train generator G
        G.zero_grad()

        G_result = G(x_)
        D_result = D(x_, G_result).squeeze()

        G_train_loss = BCE_Loss(D_result, torch.Tensor(torch.ones(D_result.size())).cuda()) + opt.L1_lambda * L1_Loss(G_result, y_)
        G_train_loss.backward()
        G_optimizer.step()

        #train_hist['G_losses'].append(G_train_loss.data[0])
        print("Generator loss: ", G_train_loss.data)
        G_losses.append(G_train_loss.data)

        num_iter += 1

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), opt.train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),
                                                              torch.mean(torch.FloatTensor(G_losses))))
    fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)
print('total training time: ', total_ptime)
