'''train a network. '''
import os
import pickle as pkl
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import network
from prep_data import get_dataset
from utils import EarlyStopping
import attack_func


CLIP_GRAD_NORM = 5.


def train(args, model, device, train_loader, optimizer, epoch, writer=None):
    model.train()
    correct, total = 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        if torch.sum(torch.isnan(loss)) > 0:
            raise ValueError('nan loss')
        loss.backward()
        optimizer.step()

        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()), flush=True)
            if writer is not None:
                writer.add_scalar('loss', loss.item(), (epoch-1) * len(train_loader.dataset) + batch_idx)
    print('train accuracy: {:.4f}'.format(100. * correct / total))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))
    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch classification example')
    parser.add_argument('--dataset', type=str, help='dataset', 
                        choices=['mnist', 'usps', 'svhn', 'syn_digits', 
                                 'imagenet32x32',
                                 'cifar10', 'stl10',
                                ])
    parser.add_argument('--arch', type=str, help='network architecture')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--val_ratio', type=float, default=0.0, 
                        help='sampling ratio of validation data')
    parser.add_argument('--train_ratio', type=float, default=1.0, 
                        help='sampling ratio of training data')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--wd', type=float, default=1e-6, 
                        help='weight_decay (default: 1e-6)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output_path', type=str, help='path to save ckpt and log. ')
    parser.add_argument('--resume', type=str, help='resume training from ckpt path')
    parser.add_argument('--ckpt_file', type=str, help='init model from ckpt. ')
    parser.add_argument('--exclude_vars', type=str, 
                        help='prefix of variables not restored form ckpt, seperated with commas; valid if ckpt_file is not None')
    parser.add_argument('--imagenet_pretrain', action='store_true', help='use pretrained imagenet model')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    writer = SummaryWriter(args.output_path)

    use_normalize = True
    if args.dataset == 'imagenet32x32':
        n_classes = 1000
        args.batch_size = 256
    elif args.dataset in ["cifar10", "stl10"]:
        n_classes = 9
    elif args.dataset in ["usps", "mnist", "svhn", 'syn_digits']:
        n_classes = 10
    else:
        raise ValueError('invalid dataset option: {}'.format(args.dataset))

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    assert(args.val_ratio >= 0. and args.val_ratio < 1.)
    assert(args.train_ratio > 0. and args.train_ratio <= 1.)
    train_ds = get_dataset(args.dataset, 'train', use_normalize=use_normalize, test_size=args.val_ratio, train_size=args.train_ratio)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        get_dataset(args.dataset, 'test', use_normalize=use_normalize, test_size=args.val_ratio, train_size=args.train_ratio),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    if args.val_ratio == 0.0:
        val_loader = test_loader
    else:
        val_ds = get_dataset(args.dataset, 'val', use_normalize=use_normalize, test_size=args.val_ratio, train_size=args.train_ratio)
        val_loader = torch.utils.data.DataLoader(val_ds,
            batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.arch == "DTN":
        model = network.DTN().to(device)
    elif args.arch == 'wrn':
        model = network.WideResNet(depth=28, num_classes=n_classes, widen_factor=10, dropRate=0.0).to(device)
    else:
        raise ValueError('invalid network architecture {}'.format(args.arch))

    if args.ckpt_file is not None:
        print('initialize model parameters from {}'.format(args.ckpt_file))
        model.restore_from_ckpt(torch.load(args.ckpt_file, map_location='cpu'), 
            exclude_vars=args.exclude_vars.split(',') if args.exclude_vars is not None else [])
        print('accuracy on test set before fine-tuning')
        test(args, model, device, test_loader)
    
    if args.resume is not None:
        assert(os.path.isfile(args.resume))
        print('resume training from {}'.format(args.resume))
        model.load_state_dict(torch.load(args.resume))

    if use_cuda:
        # model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    if args.dataset.startswith("cifar") or args.dataset in ['stl10']:
        lr_decay_step = 100
        lr_decay_rate = 0.1
        PATIENCE = 100
        optimizer = optim.SGD(model.get_parameters(args.lr), momentum=args.momentum, weight_decay=args.wd)
        scheduler = MultiStepLR(optimizer, milestones=[150,250], gamma=0.1)
    elif args.dataset in ["mnist", "usps", "svhn", "syn_digits"]:
        lr_decay_step = 50
        lr_decay_rate = 0.5
        if args.dataset == 'svhn':
            PATIENCE = 10
        else:
            PATIENCE = 50
        optimizer = optim.SGD(model.get_parameters(args.lr), momentum=0.5, weight_decay=args.wd)
        scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
    elif args.dataset == 'imagenet32x32':
        PATIENCE = 10
        lr_decay_step = 10
        lr_decay_rate = 0.2
        optimizer = torch.optim.SGD(
            model.get_parameters(args.lr), momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_rate)
    else:
        raise ValueError("invalid dataset option: {}".format(args.dataset))


    early_stop_engine = EarlyStopping(PATIENCE)

    print("args:{}".format(args))

    # start training. 
    best_accuracy = 0.
    save_path = os.path.join(args.output_path, "model.pt")
    time_stats = []
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train(args, model, device, train_loader, optimizer, epoch, writer)
        training_time = time.time() - start_time
        print('epoch: {} training time: {:.2f}'.format(epoch, training_time))
        time_stats.append(training_time)

        val_accuracy = test(args, model, device, val_loader)
        scheduler.step()

        writer.add_scalar("val_accuracy", val_accuracy, epoch)
        if val_accuracy >= best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
        
        if epoch % 20 == 0:
            print('accuracy on test set at epoch {}'.format(epoch))
            test(args, model, device, test_loader)

        if early_stop_engine.is_stop_training(val_accuracy):
            print("no improvement after {}, stop training at epoch {}\n".format(
                PATIENCE, epoch))
            break
    
    # print('finish training {} epochs'.format(args.epochs))
    mean_training_time = np.mean(np.array(time_stats))
    print('Average training_time: {}'.format(mean_training_time))
    print('load ckpt with best validation accuracy from {}'.format(save_path))
    model.load_state_dict(torch.load(save_path, map_location='cpu'))
    test_accuracy = test(args, model, device, test_loader)

    writer.add_scalar("test_accuracy", test_accuracy, args.epochs)
    with open(os.path.join(args.output_path, 'accuracy.pkl'), 'wb') as pkl_file:
        pkl.dump({'train': best_accuracy, 'test': test_accuracy, 
                  'training_time': mean_training_time}, pkl_file)


if __name__ == '__main__':
    main()
