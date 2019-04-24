import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

from wide_resnet.model import wide_resnet


def make_model(num_gpus=1):
	net = wide_resnet()
	net = net.to('cuda')
	net = torch.nn.DataParallel(net, device_ids=list(range(num_gpus)))
	cudnn.benchmark = True

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.1,
						  momentum=0.9, weight_decay=5e-4)
	step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
	return net, criterion, optimizer, step_lr_scheduler
