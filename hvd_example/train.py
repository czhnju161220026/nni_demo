'''
Training script for CIFAR-10/100
Copyright (c) Wei YANG, 2017
'''
from __future__ import print_function

import argparse
import os
import shutil
import time
import datetime
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
from torch.multiprocessing import Process
import horovod.torch as hvd
from model import ResNet
import nni
from math import floor

from utils import AverageMeter, accuracy, DataPartitioner



# Validate dataset

# Use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
seed = random.randint(1,10000)
random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

# GPU allocation using Horovod
hvd.init()
local_rank = hvd.local_rank()
torch.cuda.set_device(local_rank)
world_size = hvd.size()
print('worldsize: %d' % world_size)


def main():
    global param
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing dataset %s' % 'cifar10')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataloader = datasets.CIFAR10
    num_classes = 10
    


    trainset = dataloader(root="/data/nfs/cifar", train=True, download=True, transform=transform_train)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset,num_replicas=hvd.size(), rank=hvd.rank())
    trainloader = data.DataLoader(dataset=trainset, batch_size=param['batch_size'] * world_size, shuffle=False, sampler=sampler)

    testset = dataloader(root="/data/nfs/cifar", train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=param['batch_size'] * world_size, shuffle=False, num_workers=2)

    # Model
    print("==> creating model '{}'".format("Resnet"))
    model = ResNet(depth=50, num_classes=num_classes)

    device = torch.device('cuda', local_rank)
    model = model.to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)    
    print('Model on cuda:%d' % local_rank)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=param['lr'], momentum=0.9, weight_decay=5e-4)
    # 用horovod封装优化器
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    # 广播参数
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Train and val
    result = 0
    for epoch in range(start_epoch, param['epoch']):
        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, epoch, use_cuda)
        print('Rank:{} Epoch[{}/{}]: LR: {:.3f}, Train loss: {:.5f}, Test loss: {:.5f}, Train acc: {:.2f}, Test acc: {:.2f}.'.format(local_rank,epoch+1, param['epoch'], param['lr'], 
        train_loss, test_loss, train_acc, test_acc))
        result = test_acc
        nni.report_intermediate_result(result)
    if local_rank == 0:
        nni.report_final_result(result)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
       
        if use_cuda:
            inputs, targets = inputs.cuda(local_rank), targets.cuda(local_rank, async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(local_rank), targets.cuda(local_rank)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
    return (losses.avg, top1.avg)



def adjust_learning_rate(optimizer, epoch):
    global schedule
    global param
    if epoch in schedule:
        param['lr'] *= 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    global param
    global schedule
    start = datetime.datetime.now()
    param = nni.get_next_parameter()
    # param = {'epoch':64, 'lr':0.01, 'wd':1e-4, 'batch_size':16}
    schedule = []
    epoch = param['epoch']
    schedule.append(floor(epoch / 3 * 2))
    schedule.append(floor(epoch / 4 * 3))
    print("Prepare param: %r" % param)
    print("Set schedule point: %r" % schedule)
    main()
    end = datetime.datetime.now()
    delta = end-start
    print("Time elapsed: %d s." % delta.seconds)
