# This module is adapted from https://github.com/mahyarnajibi/FreeAdversarialTraining/blob/master/main_free.py
# Which in turn was adapted from https://github.com/pytorch/examples/blob/master/imagenet/main.py
import init_paths
import argparse
import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import math
import numpy as np
from utils import *
from validation import validate
#import torchvision.models as models
import models
from models.imagenet_resnet import BasicBlock, Bottleneck
from multiprocessing import Pool
#from torchvision.models.resnet import BasicBlock, Bottleneck
import pdb
import wandb
import random


from apex import amp
import copy


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    parser.add_argument('--output_prefix',
                        default='fast_adv',
                        type=str,
                        help='prefix used to define output path')
    parser.add_argument('-c',
                        '--config',
                        default='configs.yml',
                        type=str,
                        metavar='Path',
                        help='path to the config file (default: configs.yml)')
    parser.add_argument('--resume',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e',
                        '--evaluate',
                        dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained',
                        dest='pretrained',
                        action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--restarts', default=1, type=int)
    return parser.parse_args()


# Parase config file and initiate logging
configs = parse_config_file(parse_args())
logger = initiate_logger(configs.output_name, configs.evaluate)
print = logger.info
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()
criterion_batch = nn.CrossEntropyLoss(reduction='none').cuda()


def main():
    torch.autograd.set_detect_anomaly(True)
    # Scale and initialize the parameters
    best_prec1 = 0
    
    wandb.init(project="ImangeNet CutMix 100epoch", entity="neonsign", name = configs.output_name)
    wandb.config.update(configs)

    # Create output folder
    if not os.path.isdir(os.path.join('trained_models', configs.output_name)):
        os.makedirs(os.path.join('trained_models', configs.output_name))

    # Log the config details
    logger.info(pad_str(' ARGUMENTS '))
    for k, v in configs.items():
        print('{}: {}'.format(k, v))
    logger.info(pad_str(''))

    # Create the model
    if configs.pretrained:
        print("=> using pre-trained model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(configs.TRAIN.arch))
        model = models.__dict__[configs.TRAIN.arch]()

    def init_dist_weights(model):
        for m in model.modules():
            if isinstance(m, BasicBlock):
                m.bn2.weight = nn.Parameter(torch.zeros_like(m.bn2.weight))
            if isinstance(m, Bottleneck):
                m.bn3.weight = nn.Parameter(torch.zeros_like(m.bn3.weight))
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    init_dist_weights(model)

    # Wrap the model into DataParallel
    model.cuda()

    # reverse mapping
    param_to_moduleName = {}
    for m in model.modules():
        for p in m.parameters(recurse=False):
            param_to_moduleName[p] = str(type(m).__name__)

    group_decay = [p for p in model.parameters() if 'BatchNorm' not in param_to_moduleName[p]]
    group_no_decay = [p for p in model.parameters() if 'BatchNorm' in param_to_moduleName[p]]
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=0)]
    optimizer = torch.optim.SGD(groups,
                                0,
                                momentum=configs.TRAIN.momentum,
                                weight_decay=configs.TRAIN.weight_decay)

#     if configs.TRAIN.clean_lam > 0 and not configs.evaluate:
#         model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1024)
    model = torch.nn.DataParallel(model)

    # Resume if a valid checkpoint path is provided
    if configs.resume:
        if os.path.isfile(configs.resume):
            print("=> loading checkpoint '{}'".format(configs.resume))
            checkpoint = torch.load(configs.resume)
            configs.TRAIN.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(configs.resume,
                                                                checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs.resume))

    # Initiate data loaders
    traindir = os.path.join(configs.data, 'train')
    valdir = os.path.join(configs.data, 'val')
        
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(configs.DATA.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(traindir, train_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=configs.DATA.batch_size,
                                               shuffle=True,
                                               num_workers=configs.DATA.workers,
                                               pin_memory=True,
                                               sampler=None,
                                               drop_last=True)

    val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir, test_transform),
                                             batch_size=configs.DATA.batch_size,
                                             shuffle=False,
                                             num_workers=configs.DATA.workers,
                                             pin_memory=True,
                                             drop_last=False)

    # If in evaluate mode: perform validation on PGD attacks as well as clean samples
    if configs.evaluate:
        validate(val_loader, model, criterion, configs, logger)
        return

    lr_schedule = lambda t: np.interp([t], configs.TRAIN.lr_epochs, configs.TRAIN.lr_values)[0]

    if configs.TRAIN.mp > 0:
        mp = Pool(configs.TRAIN.mp)
    else:
        mp = None

    wandb.watch(model)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        # train for one epoch
        tprec1, tprec5, tloss, lr = train(scaler, train_loader, model, optimizer, epoch, lr_schedule, configs.TRAIN.clean_lam, mp=mp)

        # evaluate on validation set
        prec1, prec5, loss = validate(val_loader, model, criterion, configs, logger)
        
        wandb.log({
            "Loss/train": tloss, 
            "Loss/validation": loss, 
            "prec1/train": tprec1, 
            "prec1/validation": prec1, 
            "prec5/train": tprec5, 
            "prec5/validation": prec5, 
            "Learning Rate": lr})
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if is_best:
            wandb.run.summary["best_prec1"] = best_prec1
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'arch': configs.TRAIN.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, os.path.join('trained_models', f'{configs.output_name}'), epoch + 1)


def train(scaler, train_loader, model, optimizer, epoch, lr_schedule, clean_lam=0, mp=None):
    max_norm = 5
    mean = torch.Tensor(np.array(configs.TRAIN.mean)[:, np.newaxis, np.newaxis])
    mean = mean.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()
    std = torch.Tensor(np.array(configs.TRAIN.std)[:, np.newaxis, np.newaxis])
    std = std.expand(3, configs.DATA.crop_size, configs.DATA.crop_size).cuda()

    # Initialize the meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)

        target = target.cuda(non_blocking=True)
        data_time.update(time.time() - end)

        # update learning rate
        lr = lr_schedule(epoch + (i + 1) / len(train_loader))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad()

        input.sub_(mean).div_(std)
        lam = np.random.beta(configs.TRAIN.alpha, configs.TRAIN.alpha)

#         #DropMix
#         ignore_index = torch.randperm(len(target))[:round(len(target) * configs.TRAIN.ratio)]
#         ignore_mask = torch.zeros(len(target)).bool()
#         ignore_mask[ignore_index] = True

#         target_mixup = target[~ignore_mask]
#         target_ignore = target[ignore_mask]
#         input_mixup = input[~ignore_mask]
#         input_ignore = input[ignore_mask]

#         #CutMix
#         rand_index = torch.randperm(input_mixup.size()[0]).cuda()
#         target_a = torch.cat((target_mixup, target_ignore), 0)
#         target_b = torch.cat((target_mixup[rand_index], target_ignore), 0)
#         bbx1, bby1, bbx2, bby2 = rand_bbox(input_mixup.size(), lam)
#         input_mixup[:, :, bbx1:bbx2, bby1:bby2] = input_mixup[rand_index, :, bbx1:bbx2, bby1:bby2]
#         # adjust lambda to exactly match pixel ratio
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_mixup.size()[-1] * input_mixup.size()[-2]))
#         input = torch.cat((input_mixup, input_ignore),0)
#         ###
# #         pdb.set_trace()

#         #Mixup
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        input_mixup = input * lam + input[rand_index] * (1. - lam)

        input_var = Variable(input_mixup, requires_grad=True)

        if clean_lam == 0:
            model.eval()

        output = model(input_var)
        if torch.any(torch.isnan(output)):
            pdb.set_trace()
#         loss_clean = criterion(output, target)
#         #cutmix
        loss_clean = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

#         if clean_lam > 0:
#             with amp.scale_loss(loss_clean, optimizer) as scaled_loss:
#                 scaled_loss.backward()
#         else:
#             loss_clean.backward()
        loss_clean.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
#         scaler.scale(loss_clean).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         if clean_lam > 0:
#             with torch.cuda.amp.autocast(): 
#                 output = model(input_var)
#         #         loss_clean = criterion(output, target)
#         #         #cutmix
#                 loss_clean = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
#             scaler.scale(loss_clean).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             output = model(input_var)
#     #         loss_clean = criterion(output, target)
#     #         #cutmix
#             loss_clean = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
#             loss_clean.backward()
#             optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        losses.update(loss_clean.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % configs.TRAIN.print_freq == 0:
            print('Train Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR {lr:.3f}'.format(epoch,
                                       i,
                                       len(train_loader),
                                       batch_time=batch_time,
                                       data_time=data_time,
                                       top1=top1,
                                       top5=top5,
                                       cls_loss=losses,
                                       lr=lr))
            sys.stdout.flush()
    print(' Epoch {epoch}  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(epoch=epoch, top1=top1, top5=top5))
    return top1.avg, top5.avg, losses.avg, lr

if __name__ == '__main__':
    main()