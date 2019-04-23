import itertools
import os
import time

import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from model import densenet_cifar
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batches', type=int, default=None)
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch_size', default=128, help='')
parser.add_argument('--num_worker', default=4, help='')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('==> Preparing data..')
transforms_train = transforms.Compose([
	transforms.RandomCrop(32, padding=4),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transforms_test = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset_train = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms_train)
dataset_test = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transforms_test)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, 
	                      shuffle=True, num_workers=args.num_worker)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=100, 
	                     shuffle=False, num_workers=args.num_worker)

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
	       'dog', 'frog', 'horse', 'ship', 'truck')


def make_model(num_gpus=1):
	global net, criterion, optimizer, step_lr_scheduler
	print('==> Making model..')
	net = densenet_cifar()
	net = net.to(device)
	if device == 'cuda':
		net = torch.nn.DataParallel(net, device_ids=list(range(num_gpus)))
		cudnn.benchmark = True

	if args.resume is not None:
		checkpoint = torch.load('./save_model/' + args.resume)
		net.load_state_dict(checkpoint['net'])

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.1,
						  momentum=0.9, weight_decay=1e-4)
	step_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225], gamma=0.1)


def train(epoch):
	global net, criterion, optimizer, step_lr_scheduler
	net.train()

	train_loss = 0
	correct = 0
	total = 0

	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if args.batches is not None and batch_idx == args.batches:
			break
		inputs = inputs.to(device)
		targets = targets.to(device)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)
		correct += predicted.eq(targets).sum().item()
		if batch_idx % 10 == 0:
			print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx, 
				  len(train_loader), train_loss/(batch_idx+1), 100.*correct/total))



def test(epoch, best_acc):
	global net, criterion, optimizer, step_lr_scheduler
	net.eval()

	test_loss = 0
	correct = 0
	total = 0

	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()
			
			if batch_idx % 10 == 0:
				print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx, 
				  len(test_loader), test_loss/(batch_idx+1), 100 * correct/total))

	acc = 100 * correct / total

	if acc > best_acc:
		print('==> Saving model..')
		state = {
		    'net': net.state_dict(),
		    'acc': acc,
		    'epoch': epoch,
		}
		if not os.path.isdir('save_model'):
		    os.mkdir('save_model')
		torch.save(state, './save_model/ckpt.pth')
		best_acc = acc

	return best_acc


def main():
	best_acc = 0
	if args.resume is not None:
		test(epoch=0, best_acc=0)
	else:
		for epoch in range(args.epochs):
			step_lr_scheduler.step()
			train(epoch)
			best_acc = test(epoch, best_acc)
			print('best test accuracy is ', best_acc)
		return best_acc


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


if __name__ == '__main__':
	if args.benchmark:
		device_count = torch.cuda.device_count()
		print(f'Found {device_count} CUDA devices.')
		dev_count_range = range(1, device_count+1)
		durations = {i: [] for i in dev_count_range}
		accuracies = {i: [] for i in dev_count_range}
		for num_gpus in ncycles(dev_count_range, n=4):
			t0 = time.time()
			make_model(num_gpus=num_gpus)
			accu = main()
			t1 = time.time()

			durations[num_gpus].append(t1 - t0)
			accuracies[num_gpus].append(accu)
		print('---- BENCHMARK PARAMS ----')
		print(f'Num epochs  {args.epochs}')
		print(f'Num batches {args.batches}')
		print('---- BENCHMARK RESULT ----')
		for i in dev_count_range:
			time_mean = torch.tensor(durations[i]).mean().item()
			time_std = torch.tensor(durations[i]).std().item()
			accu_mean = torch.tensor(accuracies[i]).mean().item()
			accu_std = torch.tensor(accuracies[i]).std().item()
			print(f'{i} gpu(s)')
			print('    Time: Mean    {:>4.0f}s  Std   {:>4.2f}s'.format(time_mean, time_std))
			print('    Accu: Mean {:2.4f}%  Std {:1.4f}%'.format(accu_mean, accu_std))
	else:
		main()
