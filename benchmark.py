import argparse
import itertools
import os
import time

import torch.cuda
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from train import train_epochs
import densenet.train
import wide_resnet.train


model_factories = {
    'densenet': densenet.train.make_model,
    'wideresnet': wide_resnet.train.make_model
}

parser = argparse.ArgumentParser(description='Image classification speed benchmark')
parser.add_argument('model', choices=model_factories.keys(), type=str)
parser.add_argument('--measurements', type=int, default=4, help='Num measurements for avg and std')
parser.add_argument('--size', type=int, default=2, help='image size multiplier')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batches', type=int, default=None, help='stop early for testing')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size PER GPU')
args = parser.parse_args()


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def print_row(*fields):
    print(','.join(map(str, fields)))


if __name__ == '__main__':
    transforms_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32 * args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transforms_test = transforms.Compose([
        transforms.Resize(32 * args.size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    cifar_path = os.environ.get('CIFAR', './data')
    dataset_train = torchvision.datasets.CIFAR10(root=cifar_path, train=True, download=True, transform=transforms_train)
    dataset_test = torchvision.datasets.CIFAR10(root=cifar_path, train=False, download=True, transform=transforms_test)
    datasets = dataset_train, dataset_test

    make_model = model_factories[args.model]

    device_count = torch.cuda.device_count()
    print(f'Found {device_count} CUDA devices.')
    dev_count_range = range(1, device_count + 1)
    durations = {i: [] for i in dev_count_range}
    accuracies = {i: [] for i in dev_count_range}
    n_m = args.measurements
    for num_gpus in tqdm(ncycles(dev_count_range, n=n_m), total=n_m * device_count, desc='Performing benchmark'):
        t0 = time.time()
        accu = train_epochs(datasets, make_model, args.epochs, num_gpus, batch_size=args.batch_size * num_gpus,
                            n_batches=args.batches)
        t1 = time.time()

        durations[num_gpus].append(t1 - t0)
        accuracies[num_gpus].append(accu)
    print('---- BENCHMARK PARAMS ----')
    print(f'Num epochs    {args.epochs}')
    print(f'Num batches   {args.batches}')
    print(f'Batch size    {args.batch_size}')
    print(f'Img size mul  {args.size}')
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
