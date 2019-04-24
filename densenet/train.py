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

from tqdm import tqdm

parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--benchmark', action='store_true')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batches', type=int, default=None)
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument('--batch-size', type=int, default=128, help='')
parser.add_argument('--num_worker', type=int, default=4, help='')
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

# there are 10 classes so the dataset name is cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def make_loaders(batch_size):
    global train_loader, test_loader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True, num_workers=args.num_worker)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                              shuffle=False, num_workers=args.num_worker)


def make_model(num_gpus=1):
    global net, criterion, optimizer, step_lr_scheduler
    # print('==> Making model..')
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
    # if batch_idx % 10 == 0:
    # 	print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx,
    # 		  len(train_loader), train_loss/(batch_idx+1), 100.*correct/total))


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

        # if batch_idx % 10 == 0:
        # 	print('epoch : {} [{}/{}]| loss: {:.3f} | acc: {:.3f}'.format(epoch, batch_idx,
        # 	  len(test_loader), test_loss/(batch_idx+1), 100 * correct/total))

    acc = 100 * correct / total

    if acc > best_acc:
        # print('==> Saving model..')
        # state = {
        #     'net': net.state_dict(),
        #     'acc': acc,
        #     'epoch': epoch,
        # }
        # if not os.path.isdir('save_model'):
        #     os.mkdir('save_model')
        # torch.save(state, './save_model/ckpt.pth')
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
        # print('best test accuracy is ', best_acc)
        return best_acc


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def print_row(*fields):
    print('\t'.join(map(str, fields)))


if __name__ == '__main__':
    if args.benchmark:
        device_count = torch.cuda.device_count()
        print(f'Found {device_count} CUDA devices.')
        dev_count_range = range(1, device_count + 1)
        durations = {i: [] for i in dev_count_range}
        accuracies = {i: [] for i in dev_count_range}
        for num_gpus in tqdm(ncycles(dev_count_range, n=4), total=4 * device_count, desc='Performing benchmark'):
            t0 = time.time()
            # RUN TRAINING
            make_loaders(args.batch_size * num_gpus)
            make_model(num_gpus=num_gpus)
            accu = main()
            t1 = time.time()

            durations[num_gpus].append(t1 - t0)
            accuracies[num_gpus].append(accu)
        print('---- BENCHMARK PARAMS ----')
        print(f'Num epochs  {args.epochs}')
        print(f'Num batches {args.batches}')
        print(f'Batch size  {args.batch_size}')
        print('---- BENCHMARK RESULT ----')
        print_row('ngpus', 'time_mean (s)', 'time_std (s)', 'accu_mean (%)', 'accu_std (%)')
        for i in dev_count_range:
            durations_arr = torch.tensor(durations[i])
            accuracies_arr = torch.tensor(accuracies[i])
            time_mean = '{:>4.0f}'.format(durations_arr.mean().item())
            time_std = '{:>4.2f}'.format(durations_arr.std().item())
            accu_mean = '{:2.4f}'.format(accuracies_arr.mean().item())
            accu_std = '{:1.4f}'.format(accuracies_arr.std().item())
            print_row(i, time_mean, time_std, accu_mean, accu_std)
    else:
        main()
