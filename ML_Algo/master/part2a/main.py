import os
import torch
import json
import copy
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
import argparse
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import time

device = "cpu"
torch.set_num_threads(4)

batch_size = 64 # batch for one node
def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """

    start_time = time.time()
    log_iter_start_time = time.time()

    model.train()
    g_list = dist.new_group([0, 1, 2, 3])
    # remember to exit the train loop at end of the epoch

    with open(f'output/{log_file_name}', 'a+') as f:
        for batch_idx, (data, target) in enumerate(train_loader):
            # Your code goes here!
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            model_out = model(data)
            loss = criterion(model_out, target)
            loss.backward()

            for param in model.parameters():
                gradents = []
                for _ in range(4):
                    gradents.append(torch.zeros_like(param.grad))
                dist.gather(param.grad, gradents, group = g_list, async_op = False)
                grad_total = torch.zeros_like(param.grad)
                for i in range(4):
                    grad_total += gradents[i]
                grad_scatter = []
                for _ in range(4):
                    grad_scatter.append(grad_total / 4)
                dist.scatter(param.grad, gradents, group = g_list, src = 0, async_op = False)

            optimizer.step()

            elapsed_time = time.time() - start_time
            f.write(f"{epoch},{batch_count},{elapsed_time}\n")

            start_time = time.time()
            if batch_idx % 20 == 0:
                log_iter_elapsed_time = time.time() - log_iter_start_time
                print(batch_idx, "loss: ", loss.item(), "\t elapsed time: {:.3f}", log_iter_elapsed_time)
            
            log_iter_start_time = time.time()

def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--master-ip', dest='master_ip', type=str)
    parser.add_argument('--num-nodes', dest='num_nodes', type=int)
    parser.add_argument('--rank', dest='rank', type=int)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_ip
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group('gloo', rank=args.rank, world_size=args.num_nodes)

    global log_file_name
    log_file_name = f"timelog.csv"
    with open(f'output/{log_file_name}', 'w+') as f:
        f.write("epoch,iteration,elpased_time\n")

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])
    training_set = datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
    train_sampler = DistributedSampler(training_set)
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=train_sampler,
                                                    shuffle=False,
                                                    pin_memory=True)
    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    training_criterion = torch.nn.CrossEntropyLoss().to(device)

    model = mdl.VGG11()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)
    # running training for one epoch
    for epoch in range(1):
        train_model(model, train_loader, optimizer, training_criterion, epoch)
        test_model(model, test_loader, training_criterion)

if __name__ == "__main__":
    torch.manual_seed(123)
    np.random.seed(123)
    main()
