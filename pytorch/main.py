from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #restrict to a single GPU
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable, grad
import numpy as np

### load project files
import models2
from models2 import weights_init

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=300000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , type=int, default=1, help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outDir', required=True, help='folder to output images and model checkpoints')
#parser.add_argument('--model', required=True, help='DCGAN | RESNET | IGAN | DRAGAN | BEGAN')
parser.add_argument('--d_labelSmooth', type=float, default=0.1, help='for D, use soft label "1-labelSmooth" for real samples')
parser.add_argument('--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D')
parser.add_argument('--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G')
parser.add_argument('--white_noise'  , type=int, default=1, help='Add white noise to inputs of discriminator to stabilize training')
parser.add_argument('--lr_decay_every', type=int, default=3000, help='decay lr this many iterations')
parser.add_argument('--save_step', type=int, default=10000, help='save weights every 50000 iterations ')
parser.add_argument('--percent', type=float, default=.5)


opt = parser.parse_args()
# opt = parser.parse_args(arg_list)
print(opt)

# Make directories
opt.outDir = './results/' + opt.outDir
opt.modelsDir = opt.outDir + '/models'
opt.imDir = opt.outDir + '/images'

# Recursively create image and model directory
try:
    os.makedirs(opt.imDir)
except OSError:
    pass
try:
    os.makedirs(opt.modelsDir)
except OSError:
    pass

opt.manualSeed = random.randint(1,10000) # fix seed, a scalar
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    
BS_real = 127
BS_fake = 64
opt.batchSize = BS_real
nc = 3
ngpu = opt.ngpu
nz = opt.nz
ngf = opt.ngf
ndf = opt.ndf
n_extra_d = opt.n_extra_layers_d
n_extra_g = opt.n_extra_layers_g
percent = opt.percent
history = torch.Tensor()

dataset = dset.ImageFolder(
    root=opt.dataRoot,
    transform=transforms.Compose([
            transforms.Resize(opt.imageSize),
            transforms.CenterCrop(opt.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
        ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=opt.workers)
loader = iter(dataloader)

# load models 
netG = models2._netG_1(ngpu, nz, nc, ngf, n_extra_g, 1, opt.imageSize)
netD = models2._netD_1(ngpu, nz, nc, ndf, n_extra_d, opt.imageSize)

netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
additive_noise = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
epsilon = 1e-20

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label, additive_noise = input.cuda(), label.cuda(), additive_noise.cuda()
    noise = noise.cuda()
    
input = Variable(input)
label = Variable(label)
additive_noise = Variable(additive_noise)
noise = Variable(noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

for iteration in range(1, opt.niter+1):
    try: 
        data = loader.next()
    except StopIteration:
        loader = iter(dataloader)
        data = loader.next()

    start_iter = time.time()
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    netD.zero_grad()
    real_cpu, _ = data
    batchSize = real_cpu.size(0)
    input.data.resize_(real_cpu.size()).copy_(real_cpu)
    label.data.resize_(real_cpu.size(0)).fill_(real_label - opt.d_labelSmooth) 

    if opt.white_noise:
        additive_noise.data.resize_(input.size()).normal_(0, 0.005)
        input.data.add_(additive_noise.data)

    output = netD(input)

    # Prevent numerical instability
    output = torch.clamp(output, min=epsilon)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.data.mean()

    # train with fake
    noise.data.resize_(BS_fake, nz, 1, 1)
    noise.data.normal_(0, 1)
    fake_o = netG(noise)

    # save to history
    if not history.size() or percent == 0:
        history = fake_o
    else:
        # permute history
        history = history[torch.randperm(history.size(0)).cuda()]
        # save percentage
        history = history[:int(percent * history.size(0)),:]
        history = torch.cat([history, fake_o])

    label.data.resize_(history.size(0)).fill_(fake_label) 

    if opt.white_noise:
        additive_noise.data.resize_(history.size()).normal_(0, 0.005)
        fake = history + additive_noise
    else:
        fake = history

    output = netD(fake.detach()) # add ".detach()" to avoid backprop through G
    output = torch.clamp(output, min=epsilon)
    errD_fake = criterion(output, label)
    errD_fake.backward() # gradients for fake/real will be accumulated
    D_G_z1 = output.data.mean()
    errD = errD_real + errD_fake

    optimizerD.step() # .step() can be called once the gradients are computed

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.data.resize_(fake_o.size(0)).fill_(real_label) # fake labels are real for generator cost
    output = netD(fake_o)
    output = torch.clamp(output, min=epsilon)
    errG = criterion(output, label)
    errG.backward() # True if backward through the graph for the second time
    D_G_z2 = output.data.mean()
    optimizerG.step()

    end_iter = time.time()

    print('[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s, history: %d'
          % (iteration, opt.niter,
             errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter-start_iter, history.size(0)))


    if iteration % 500 == 0:
        # the first 64 samples from the mini-batch are saved.
        #vutils.save_image(real_cpu[0:64,:,:,:],
        #        '%s/real_samples_%03d_%04d.png' % (opt.imDir, epoch, i), nrow=8)
        fake = netG(noise)
        vutils.save_image(fake.data[0:64,:,:,:],
                '%s/fake_samples_epoch_%03d.png' % (opt.imDir, iteration), nrow=8, normalize=True)
    if iteration % opt.save_step == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.modelsDir, iteration))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.modelsDir, iteration))
