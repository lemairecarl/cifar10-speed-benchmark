import argparse
import itertools
import time

import torch.cuda
from tqdm import tqdm

from train import train_epochs, make_datasets
import densenet.train
import wide_resnet.train


model_factories = {
    'densenet': densenet.train.make_model,
    'wideresnet': wide_resnet.train.make_model
}

parser = argparse.ArgumentParser(description='Image classification speed benchmark')
parser.add_argument('model', choices=model_factories.keys(), type=str)
parser.add_argument('--gpus', type=int, default=None, help='Number of gpus to use. Default: all')
parser.add_argument('--progressive', action='store_true', help='Try 1 gpus, 2 gpus, 3 gpus, etc.')
parser.add_argument('--measurements', type=int, default=4, help='Num measurements for avg and std')
parser.add_argument('--size', type=int, default=2, help='image size multiplier')
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batches', type=int, default=None, help='stop early for testing')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size PER GPU')
parser.add_argument('--workers-per-gpu', type=int, default=4, help='Workers per GPU')
parser.add_argument('--output-accu', action='store_true')
args = parser.parse_args()


def ncycles(iterable, n):
    "Returns the sequence elements n times"
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def print_row(*fields):
    print(','.join(map(str, fields)))


if __name__ == '__main__':
    device_count = torch.cuda.device_count()
    print(f'Found {device_count} CUDA device(s).')
    if args.gpus is not None:
        device_count = args.gpus
        print(f'Using {device_count} device(s) as requested.')
    print(f'Progressive mode: {"ON" if args.progressive else "OFF"}')

    dev_count_range = range(1, device_count + 1) if args.progressive else [device_count]
    durations = {i: [] for i in dev_count_range}
    accuracies = {i: [] for i in dev_count_range}
    n_m = args.measurements

    make_model = model_factories[args.model]
    datasets = make_datasets(img_size_mult=args.size)

    print('Warming up...')
    train_epochs(datasets, make_model, args.epochs, device_count, batch_size=args.batch_size * device_count,
                 n_batches=10, workers_per_gpu=args.workers_per_gpu)

    print('Benchmarking...')
    for num_gpus in tqdm(ncycles(dev_count_range, n=n_m), total=n_m * len(dev_count_range), desc='Benchmarking'):
        t0 = time.time()
        accu = train_epochs(datasets, make_model, args.epochs, num_gpus, batch_size=args.batch_size * num_gpus,
                            n_batches=args.batches, workers_per_gpu=args.workers_per_gpu)
        t1 = time.time()

        durations[num_gpus].append(t1 - t0)
        accuracies[num_gpus].append(accu)

    print('---- BENCHMARK PARAMS ----')
    print(f'Num epochs    {args.epochs}')
    print(f'Num batches   {args.batches}')
    print(f'Batch size    {args.batch_size}')
    print(f'Workers/gpu   {args.workers_per_gpu}')
    print(f'Img size mul  {args.size}')
    print('---- BENCHMARK RESULT ----')
    if args.output_accu:
        print_row('ngpus', 'time_mean (s)', 'time_std (s)', 'accu_mean (%)', 'accu_std (%)')
    else:
        print_row('ngpus', 'time_mean (s)', 'time_std (s)')
    for i in dev_count_range:
        durations_arr = torch.tensor(durations[i])
        accuracies_arr = torch.tensor(accuracies[i])
        time_mean = '{:.0f}'.format(durations_arr.mean().item())
        time_std = '{:.2f}'.format(durations_arr.std().item())
        if args.output_accu:
            accu_mean = '{:2.4f}'.format(accuracies_arr.mean().item())
            accu_std = '{:1.4f}'.format(accuracies_arr.std().item())
            print_row(i, time_mean, time_std, accu_mean, accu_std)
        else:
            print_row(i, time_mean, time_std)
