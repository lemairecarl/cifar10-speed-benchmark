import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from densenet.model import densenet_cifar


def make_model(num_gpus=1):
    # print('==> Making model..')
    net = densenet_cifar()
    net = net.to('cuda')
    net = torch.nn.DataParallel(net, device_ids=list(range(num_gpus)))
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=1e-4)
    step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)
    return net, criterion, optimizer, step_lr_scheduler
