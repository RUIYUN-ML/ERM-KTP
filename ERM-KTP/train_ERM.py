#!/usr/bin/env python
# -*- coding: utf-8 -*-

print('\n'+"="*100)
from datetime import datetime

print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
import getpass
import socket

print(getpass.getuser()+'@'+socket.gethostname())
print()

import argparse
import errno
import os
import re
import sys
import time

import torch
import torch.nn as nn
from dataLoader import DataLoader
from model.resnet_20 import resnet20
from model.resnet_50 import resnet50
from model.resnext import resnext50
from seed import set_seed
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# parameters setting
parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--name', default='test_model', help='filename to output best model') #save output
parser.add_argument('--model', default='resnet20',help="models e.g. resnet20|resnet50|resnext50")
parser.add_argument('--dataset', default='cifar-10',help="datasets e.g. cifar-10|cifar-100|imagenet")
parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
parser.add_argument('--batch_size', default=256,type=int, help='batch size')
parser.add_argument('--epoch', default=200,type=int, help='epoch')
parser.add_argument('--exp_dir',default='./')
parser.add_argument('--ifmask', default='True', type=str, help="whether use learnable mask (i.e. gate matrix)")
parser.add_argument('--optim', default='sgd', type=str, help="optimizer: adam | sgd")
parser.add_argument('--lr', default=0.1, type=float, help="learning rate for normal path")
parser.add_argument('--lr_reg', default=0.1, type=float, help='lr of the loss of regularization path')
parser.add_argument('--lambda_reg', default=1e-3, type=float, help='regularization coefficient')
parser.add_argument('--warmup_epochs', default=0, type=float, help='the number of starting epochs that use mask')
parser.add_argument('--mask_period', default=3, type=int, help='how many epochs is a period of alternating STD/CSG path')
parser.add_argument('--mask_epoch_min', default=2, type=int, help='epochid % mask_period >= {this} use CSG path')
parser.add_argument('--train', default='True', type=str, help='train or test the model')
parser.add_argument('--seed', default=0, type=int, help='random seed for the entire program')
parser.add_argument('--cudnn_behavoir', default='benchmark', type=str, help='cudnn behavoir [benchmark|normal(default)|slow|none] from left to right, cudnn randomness decreases, speed decreases')
parser.add_argument('--load_checkpoint', default='', type=str, help='path to load a checkpoint')


args = parser.parse_args()
args.ifmask=True if args.ifmask == 'True' else False
args.train=True if args.train == 'True' else False

args.use_gpu = torch.cuda.is_available()
if args.use_gpu:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

print('args')
for arg in vars(args):
     print('   ',arg, '=' ,getattr(args, arg))
print()

set_seed(args.seed, args.cudnn_behavoir)



def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()

    best_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0

    L1_list = []
    CSI_list = []

    # Load unfinished model
    if args.load_checkpoint == '':
        unfinished_model_path = os.path.join(args.exp_dir, 'checkpoints/last.pt')
    else:
        unfinished_model_path = args.load_checkpoint

    if (os.path.exists(unfinished_model_path)):
        if args.train:
            print('Already exist and will continue training')
        else:
            print('Testing')
        print('loaded '+unfinished_model_path)
        checkpoint = torch.load(unfinished_model_path)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False )

            if args.train:
                epoch = checkpoint['epoch'] + 1
            else:
                epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            best_acc = checkpoint['best_acc'] if 'best_acc' in checkpoint else 0.0
            best_train_acc = checkpoint['best_train_acc'] if 'best_train_acc' in checkpoint else 0.0
            best_epoch = checkpoint['best_epoch'] if 'best_epoch' in checkpoint else 0.0
        else:
            model.load_state_dict(checkpoint, strict=False)
            epoch = args.epoch
    else:
        epoch = 0

    while epoch <= num_epochs or not args.train:
        epoch_time = time.time()

        if args.train:
            print('Epoch %3d/%3d' % (epoch, num_epochs), end=' | ')
        else:
            print('Epoch %3d' % (epoch), end=' | ')
        #each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train' and args.train:
                model.train()
            else:
                model.eval()

            ifmask = (epoch % args.mask_period >= args.mask_epoch_min and epoch >= args.warmup_epochs and args.ifmask and phase == 'train')

            running_loss = 0.0
            running_loss_0 = 0.0
            running_corrects = 0.0
            running_regulization_loss = 0.0

            # change tensor to variable(including some gradient info)
            # use variable.data to get the corresponding tensor
            for iteration, data in enumerate(dataloaders[phase]):

                inputs,labels = data
                if args.use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                #zero the parameter gradients
                optimizer.zero_grad()

                #forward
                if ifmask:
                    outputs, regulization_loss = model(inputs, labels=labels)
                    regulization_loss = regulization_loss.mean()
                    loss_0 = criterion(outputs, labels)
                    loss = loss_0 + regulization_loss * args.lambda_reg
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                preds = torch.argmax(outputs.data, 1)

                if phase == 'train' and  args.train:
                    loss.backward()
                    optimizer.step()

                if args.ifmask:
                    model.module.lmask.clip_lmask()

                y = labels.data
                batch_size = labels.data.shape[0]
                running_loss += loss.item()
                running_corrects += torch.sum(preds == y)
                if ifmask:
                    running_loss_0 += loss_0.item()
                    running_regulization_loss += (regulization_loss * args.lambda_reg).item()

            if phase == 'train' and  args.train:
                    scheduler.step()  # (loss)

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]
            if ifmask:
                epoch_loss_0 = running_loss_0 / dataset_sizes[phase]
                epoch_regulization_loss = running_regulization_loss / dataset_sizes[phase]
            else:
                epoch_loss_0 = 0.0
                epoch_regulization_loss = 0.0

            if phase == 'train':
                if best_train_acc < epoch_acc:
                    best_train_acc = epoch_acc

            if phase == 'val':
                if args.ifmask:
                    mask_density = model.module.lmask.get_density()
                    mask_csi = model.module.lmask.get_CSI()

                    L1_list.append(mask_density.item())
                    CSI_list.append(mask_csi.item())


            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model = model.state_dict()

            cost_time = time.time() - epoch_time


            print('%5s %2dm%2ds Acc:%.4f Loss:%.4f' %
                (phase, cost_time // 60, cost_time % 60, epoch_acc, epoch_loss), end=' ')

            if args.ifmask:
                if phase == 'train':
                    print('LMain:%.4f LossReg:%.4f' %
                        (epoch_loss_0, epoch_regulization_loss), end=' ')
                if phase == 'val':
                    print('MaskDens:%.4f' %
                        (mask_density), end=' ')
                    print('MaskCSI:%.4f' %
                        (mask_csi), end=' ')
            print('|', end=' ')

            if phase == 'val':
                print('Best epoch:%-3d train_acc:%.4f val_acc:%.4f ' % (best_epoch, best_train_acc, best_acc), end=' ')
                #if args.ifmask == True:
                    #print(model.module.lmask.get_channel_mask())
                print()

        if not args.train:
            break


        checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
        if args.train or args.load_checkpoint == '':
            os.makedirs(checkpoint_dir, exist_ok=True) # if no such path exists, iteratively created the dir


        def update_checkpoint_link(target_link_list):
            old_target_list = []
            target_list = []
            for target_name, link_name in target_link_list:
                target_path = os.path.join(checkpoint_dir, target_name)
                link_path = os.path.join(checkpoint_dir, link_name)
                if os.path.exists(link_path):
                    old_target_path = os.path.join(checkpoint_dir, os.readlink(link_path))
                    old_target_list.append(old_target_path)
                target_list.append(target_path)
                symlink_force(target_name, link_path)

            for old_target_path in set(old_target_list):
                old_epoch = int(re.findall(r'\d+', os.path.basename(old_target_path))[0])
                if old_target_path not in target_list and old_epoch % 10 != 0 and old_epoch != num_epochs - 1:
                    os.remove(old_target_path)

        if args.train:

            checkpoint_path = os.path.join(checkpoint_dir, 'epoch_%d.pt' % epoch)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_acc': best_acc,
                'best_train_acc': best_train_acc,
                'best_epoch': best_epoch,
                'CSI':CSI_list,
                'L1':L1_list
            }, checkpoint_path)

            update_checkpoint_link([
                ('epoch_%d.pt' % best_epoch, 'best.pt'),
                ('epoch_%d.pt' % epoch, 'last.pt')])

        epoch += 1

    cost_time = time.time() - since
    print ('Training complete in {:.0f}h{:.0f}m{:.0f}s'.format( (cost_time//60)//60 , (cost_time//60)%60 ,cost_time%60))

    return model, cost_time, best_acc, best_train_acc


if __name__ == '__main__':

    loader = DataLoader(args.dataset,batch_size=args.batch_size, seed=args.seed)
    dataloaders, dataset_sizes = loader.load_data()

    num_classes = 10
    if args.dataset == 'cifar-10':
        num_classes = 10
    if args.dataset == 'cifar-100':
        num_classes = 100
    if args.dataset == 'imagenet':
        num_classes = 200

    if args.model == 'resnet50':
        model = resnet50(num_classes=num_classes, ifmask=args.ifmask, pretrained=False)
    if args.model == 'resnet20':
        model = resnet20(ifmask=args.ifmask, num_classes=num_classes)
    elif args.model == 'resnext50':
        model = resnext50(num_classes=num_classes, ifmask=args.ifmask)


    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
    else:
        print("invalid args.optim. args.optim = [sgd|adam]")
        sys.exit()

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)

    if args.use_gpu:
        model = torch.nn.DataParallel(model) # device_ids=args.gpu_ids
        model = model.cuda() # args.gpu_ids[0]
    model,cost_time,best_acc,best_train_acc = train_model(model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            scheduler=scheduler,
                                            num_epochs=args.epoch)


    
