import os

import torch
import torch.cuda
import torchvision
import torchvision.transforms as transforms


def make_datasets(img_size_mult):
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32 * img_size_mult),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(32 * img_size_mult),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_path = os.environ.get('CIFAR', './data')
    dataset_train = torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=transforms_train)
    dataset_test = torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=True, transform=transforms_test)
    return dataset_train, dataset_test


def make_loaders(datasets, batch_size):
    dataset_train, dataset_test = datasets

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                               shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                              shuffle=False, num_workers=8)
    return train_loader, test_loader


def train(net, criterion, optimizer, train_loader, n_batches=-1):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx == n_batches:
            break
        inputs = inputs.to('cuda')
        targets = targets.to('cuda')
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


def test(net, criterion, test_loader, n_batches=-1):
    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if batch_idx == n_batches:
                break
            inputs = inputs.to('cuda')
            targets = targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100 * correct / total

    return acc


def train_epochs(datasets, make_model, n_epochs, n_gpus, batch_size, n_batches=-1):
    train_loader, test_loader = make_loaders(datasets, batch_size)
    net, criterion, optimizer, step_lr_scheduler = make_model(num_gpus=n_gpus)
    for epoch in range(n_epochs):
        step_lr_scheduler.step()
        train(net, criterion, optimizer, train_loader, n_batches)
    return test(net, criterion, test_loader, n_batches)
