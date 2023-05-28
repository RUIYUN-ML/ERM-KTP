
import copy
import errno
import os
import re
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from classAcc_valid import classACC_valid
from dataLoader import UnlearningDataLoader
from model.resnet_50 import resnet50
from model.resnet_20 import resnet20
from model.resnext import resnext50
from parser_init import *
from seed import set_seed



args = parser_init()
args.train = True if args.train == 'True' else False


args.use_gpu = torch.cuda.is_available()
if args.use_gpu:
    try:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
    except ValueError:
        raise ValueError(
            'Argument --gpu_ids must be a comma-separated list of integers only')

print('args')
for arg in vars(args):
    print('   ', arg, '=', getattr(args, arg))
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


def load_model(model):
    # Load unfinished model
    if args.load_checkpoint == '':
        unfinished_model_path = os.path.join(
            args.exp_dir, 'checkpoints/last.pt')
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
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            if args.train:
                epoch = checkpoint['epoch'] + 1
                #epoch = 0
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

    return model


def load_fc(ori_model, model, num_classes, target):
    ori_checkpoint = ori_model.state_dict()
    if args.model == 'resnet20':
        layer = ['module.linear.weight', 'module.linear.bias']
    else:
        layer = ['module.fc.weight', 'module.fc.bias']

    linear = {k: v for k, v in ori_checkpoint.items() if k in [layer[0], layer[1]]}

    cmask = ori_checkpoint['module.lmask.mask']
    mask = (torch.where(cmask[:,target]>0, torch.zeros_like(cmask[:,target]), torch.ones_like(cmask[:,target])))
    if isinstance (target, list):
        mask = (torch.sum(mask, dim=1)==len(target))
    mask = mask.repeat(num_classes, 1)

    linear[layer[0]]*=mask

    linear[layer[0]][target] = 0
    linear[layer[1]][target] = 0

    stat_dict = model.state_dict()
    stat_dict.update(linear)
    model.load_state_dict(stat_dict)

    return model

def train_model(model, model_ori, criterion, optimizer, scheduler, num_epochs=200):
    since = time.time()

    best_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    epoch = 0
    model_ori = load_model(model_ori)
    model = load_fc(model_ori, model, num_classes=num_classes, target=target)

    

    while epoch <= num_epochs or not args.train:
        epoch_time = time.time()

        if args.train:
            print('Epoch %3d/%3d' % (epoch, num_epochs), end=' | ')
        else:
            print('Epoch %3d' % (epoch), end=' | ')
        # each epoch has a training and validation phase
        for phase in ['remaining']:
            if phase == 'remaining' and args.train:
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0

            # change tensor to variable(including some gradient info)
            # use variable.data to get the corresponding tensor
            for iteration, data in enumerate(dataloaders[phase]):
                inputs, labels = data
                if args.use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.no_grad():
                    origin_logit, origin_feature0, origin_feature1 = model_ori(inputs, target=target)

                logit, feature0, feature1 = model(inputs, ifmap=True)

                feature_loss0 = criterion(feature0, origin_feature0.detach()) 
                feature_loss1 = criterion(feature1, origin_feature1.detach())

                loss = feature_loss0 + feature_loss1


                if phase == 'remaining' and args.train:
                    loss.backward()
                    optimizer.step()

                preds = torch.argmax(logit.data, 1)
                y = labels.data
                batch_size = labels.data.shape[0]
                running_loss += loss.item()
                running_corrects += torch.sum(preds == y)
                
            if phase == 'remaining' and args.train and epoch > 5:
                scheduler.step()  # (loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            if phase == 'remaining':
                if best_train_acc < epoch_acc:
                    best_train_acc = epoch_acc


            cost_time = time.time() - epoch_time

            print('%5s %2dm%2ds Acc:%.4f Loss:%f' %
                  (phase, cost_time // 60, cost_time % 60, epoch_acc, epoch_loss), end=' ')

            print('|', end=' ')


        remain_train_acc, unlearn_train_acc, remain_val_acc, unlearn_val_acc = classACC_valid(model, dataloaders, num_classes, args)
        print('remain_train_acc:%-.4f unlearn_train_acc:%.4f remain_val_acc:%.4f unlearn_val_acc:%.4f' %
                (remain_train_acc, unlearn_train_acc, remain_val_acc, unlearn_val_acc), end=' ')
        print()

        if not args.train:
            break

        checkpoint_dir = os.path.join(args.exp_dir, 'checkpoints')
        if args.train or args.load_checkpoint == '':
            # if no such path exists, iteratively created the dir
            os.makedirs(checkpoint_dir, exist_ok=True)

        def update_checkpoint_link(target_link_list):
            old_target_list = []
            target_list = []
            for target_name, link_name in target_link_list:
                target_path = os.path.join(checkpoint_dir, target_name)
                link_path = os.path.join(checkpoint_dir, link_name)
                if os.path.exists(link_path):
                    old_target_path = os.path.join(
                        checkpoint_dir, os.readlink(link_path))
                    old_target_list.append(old_target_path)
                target_list.append(target_path)
                symlink_force(target_name, link_path)

            for old_target_path in set(old_target_list):
                old_epoch = int(re.findall(
                    r'\d+', os.path.basename(old_target_path))[0])
                if old_target_path not in target_list and old_epoch % 10 != 0 and old_epoch != num_epochs - 1:
                    os.remove(old_target_path)

        if args.train:

            checkpoint_path = os.path.join(
                checkpoint_dir, 'epoch_%d.pt' % epoch)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_acc': best_acc,
                'best_train_acc': best_train_acc,
                'best_epoch': best_epoch
            }, checkpoint_path)

            update_checkpoint_link([
                ('epoch_%d.pt' % best_epoch, 'best.pt'),
                ('epoch_%d.pt' % epoch, 'last.pt')])

        epoch += 1

    cost_time = time.time() - since
    print('Training complete in {:.0f}h{:.0f}m{:.0f}s'.format(
        (cost_time//60)//60, (cost_time//60) % 60, cost_time % 60))

    return model, cost_time, best_acc, best_train_acc


if __name__ == '__main__':

    loader = UnlearningDataLoader(
        args.dataset, batch_size=args.batch_size, num_unlearn=args.num_unlearn, seed=args.seed)
    dataloaders, dataset_sizes = loader.load_data()

    target = list(range(0, args.num_unlearn))

    num_classes = 10
    if args.dataset == 'cifar-10':
        num_classes = 10
    if args.dataset == 'cifar-100':
        num_classes = 100
    if args.dataset == 'imagenet':
        num_classes = 200

    if args.model == 'resnet20':
        model = resnet20(ifmask=args.ifmask, num_classes=num_classes)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=num_classes, ifmask=args.ifmask, pretrained=False)
    elif args.model == 'resnext50':
        model = resnext50(num_classes=num_classes, ifmask=args.ifmask)

    model_ori = copy.deepcopy(model)

    selected_param_names = []
    for idx, (layer_name, layer) in enumerate(model.named_children()):
        if 'linear' not in layer_name:
            for param_name, param in layer.named_parameters():
                selected_param_names.append(param)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(selected_param_names, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

    criterion = nn.MSELoss()
    scheduler = MultiStepLR(optimizer, [10,20], gamma=0.1)

    if args.use_gpu:
        model_ori, model = torch.nn.DataParallel(
            model_ori), torch.nn.DataParallel(model)  # device_ids=args.gpu_ids
        model_ori, model = model_ori.cuda(), model.cuda()

    model, cost_time, best_acc, best_train_acc = train_model(model=model, model_ori=model_ori,
                                                             optimizer=optimizer,
                                                             criterion = criterion,
                                                             scheduler=scheduler,
                                                             num_epochs=args.epoch)
