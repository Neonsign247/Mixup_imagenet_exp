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
    # Scale and initialize the parameters
    best_prec1 = 0
    
    wandb.init(project="DropMix Imagenet 100epoch", entity="neonsign", name = configs.output_name)
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

    if configs.TRAIN.clean_lam > 0 and not configs.evaluate:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", loss_scale=1024)
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
        transforms.RandomResizedCrop(configs.DATA.crop_size, scale=(configs.DATA.min_scale, 1.0)),
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
    for epoch in range(configs.TRAIN.start_epoch, configs.TRAIN.epochs):
        # train for one epoch
        tprec1, tprec5, tloss, lr = train(train_loader, model, optimizer, epoch, lr_schedule, configs.TRAIN.clean_lam, mp=mp)

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

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def train(train_loader, model, optimizer, epoch, lr_schedule, clean_lam=0, mp=None):
#     degraded_cls_list = torch.tensor([266, 193, 250, 660, 764, 620, 380, 409, 676, 887, 928, 975, 171, 311, 377, 857, 101, 46, 543, 559, 579, 803, 813, 885, 32, 748, 63, 142, 161, 186, 402, 629, 868, 878, 908, 233, 249, 304, 489, 506, 600, 696, 812, 814, 911, 923, 26, 77, 134, 202, 210, 243, 328, 382, 385, 473, 486, 522, 667, 711, 795, 797, 830, 902, 950, 68, 282, 415, 664, 838, 841, 860, 978, 999, 50, 184, 197, 383, 412, 457, 500, 507, 650, 693, 759, 778, 825, 844, 931, 939, 43, 81, 140, 165, 252, 297, 331, 345, 367, 379, 468, 546, 645, 726, 754, 756, 859, 864, 901, 972, 983, 4, 25, 115, 126, 152, 163, 204, 258, 259, 322, 442, 487, 538, 585, 589, 593, 596, 606, 617, 622, 710, 852, 914, 958, 990, 60, 73, 493, 567, 657, 848, 240, 44, 45, 71, 94, 158, 159, 178, 188, 199, 217, 288, 337, 349, 438, 466, 535, 578, 648, 678, 694, 735, 780, 786, 808, 880, 884, 980, 9, 90, 139, 182, 223, 230, 287, 295, 313, 356, 375, 448, 513, 708, 736, 740, 793, 799, 826, 856, 867, 889, 917, 918, 943, 7, 18, 23, 24, 34, 69, 74, 75, 78, 80, 82, 89, 92, 96, 97, 103, 116, 122, 131, 144, 157, 168, 174, 185, 194, 195, 213, 231, 234, 248, 261, 262, 272, 273, 274, 279, 284, 302, 307, 317, 338, 342, 343, 352, 354, 355, 368, 378, 381, 399, 420, 431, 434, 436, 437, 447, 452, 456, 460, 461, 480, 483, 503, 511, 523, 529, 530, 545, 582, 586, 626, 644, 653, 654, 692, 766, 784, 790, 800, 805, 817, 823, 827, 849, 863, 876, 892, 906, 907, 921, 936, 938, 942, 949, 960, 984, 987, 988, 992, 836, 499, 633, 88, 99, 138, 143, 181, 226, 236, 253, 283, 286, 306, 316, 318, 323, 599, 625, 671, 730, 731, 779, 816, 953]).cuda()
    degraded_cls_list = torch.tensor([
        817, 908, 285, 238, 266, 844, 836, 385, 778, 506, 526, 618, 630, 585, 409,  46, 204, 600, 516, 494, 345, 691, 159, 248, 764, 193, 578, 693, 457, 650, 587, 303, 705, 689, 493, 433, 714, 970, 265, 902, 606, 487, 878, 158, 499, 740, 838, 429, 742, 830, 931, 542, 362, 240, 676, 816, 647, 170, 515, 341, 748, 975, 848, 807, 910, 186, 273, 696, 545, 699, 459, 654, 629, 505, 617, 108, 575, 777, 99, 999, 885, 231, 146, 483, 380, 804, 484, 597, 234, 940, 786, 2,  77, 488, 653, 428, 745, 522,  25, 220, 800, 596, 546, 163, 243, 808, 928, 557, 422, 744, 390, 904, 206, 780, 756, 873, 765, 377,   7, 697, 828, 721, 729,  19, 701, 881, 215, 181, 852, 176, 120, 512, 323, 319, 264, 511, 513, 572, 121, 978, 507, 311, 731, 452, 153, 857, 262, 711, 147, 751, 962, 383, 938, 621, 182, 622, 226, 530, 381, 174, 263,  18, 864,  53, 297, 257, 884, 843, 162, 210, 917, 216, 217, 411, 103, 763, 223, 233, 191, 685, 259, 719, 811, 853,  71, 827, 797, 735, 826,  63, 188, 155, 919, 395, 397, 401, 294, 431, 580, 579, 293, 550,  78, 543,  47, 491, 391, 916, 943, 673, 287, 665, 366, 894, 445, 810, 961,  36, 461, 759, 785, 984, 327, 406, 470, 328, 430, 486, 932, 677,  68, 463, 845,  32, 187, 667,  66, 558, 623, 455, 105, 185, 879, 795, 101, 309, 882, 310, 746, 375, 739,  59, 710, 171, 280, 392, 305, 150, 671, 404, 773, 832,  27, 614, 283,  24, 439, 738, 876, 318, 813, 189, 732,  80, 326, 332, 337, 340, 349, 364, 659, 407, 418,  37, 450, 529, 518,  20, 766,  51, 770, 194,   6, 911, 144, 141, 137, 135, 123, 222, 119,  96, 992,  94,  93,  92, 993,  85, 996, 981, 251, 918, 255, 498, 771, 282,  60, 824, 841, 304, 968, 541, 274, 442, 166, 307, 886, 352, 344,   9, 950, 333, 286, 812, 814, 330, 230, 316, 741
    ]).cuda()#5 runs
    degraded_cls = degraded_cls_list[0:configs.TRAIN.ratio]
    
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
        
        #DropMix_class
        ignore_index = torch.where((target[..., None] == degraded_cls).any(-1))[0]
#         indices = torch.randperm(len(ignore_index))[:round(len(ignore_index) * args.ignore_alpha)]
#         ignore_index = ignore_index[indices]
        ignore_mask = torch.zeros(len(target)).bool()
        ignore_mask[ignore_index] = True

        target_mixup = target[~ignore_mask]
        target_ignore = target[ignore_mask]
        input_mixup = input[~ignore_mask]
        input_ignore = input[ignore_mask]
        
        #CutMix
        rand_index = torch.randperm(input_mixup.size()[0]).cuda()
        target_a = torch.cat((target_mixup, target_ignore), 0)
        target_b = torch.cat((target_mixup[rand_index], target_ignore), 0)
        bbx1, bby1, bbx2, bby2 = rand_bbox(input_mixup.size(), lam)
        input_mixup[:, :, bbx1:bbx2, bby1:bby2] = input_mixup[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_mixup.size()[-1] * input_mixup.size()[-2]))
        input = torch.cat((input_mixup, input_ignore),0)
        ###
#         pdb.set_trace()
        
        input_var = Variable(input, requires_grad=True)

        if clean_lam == 0:
            model.eval()

        output = model(input_var)
#         loss_clean = criterion(output, target)
        #cutmix
        loss_clean = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        if torch.isnan(loss_clean):
            pdb.set_trace()

        if clean_lam > 0:
            with amp.scale_loss(loss_clean, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_clean.backward()
        optimizer.step()

        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        losses.update(loss_clean.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

#         if i % configs.TRAIN.print_freq == 0:
#             print('Train Epoch: [{0}][{1}/{2}]\t'
#                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
#                   'Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
#                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
#                   'LR {lr:.3f}'.format(epoch,
#                                        i,
#                                        len(train_loader),
#                                        batch_time=batch_time,
#                                        data_time=data_time,
#                                        top1=top1,
#                                        top5=top5,
#                                        cls_loss=losses,
#                                        lr=lr))
#             sys.stdout.flush()
    print(' Epoch {epoch}  Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(epoch=epoch, top1=top1, top5=top5))
    return top1.avg, top5.avg, losses.avg, lr

if __name__ == '__main__':
    main()
