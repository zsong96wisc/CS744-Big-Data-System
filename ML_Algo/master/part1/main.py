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
import time
device = "cpu"
torch.set_num_threads(4)

batch_size = 256 # batch for one node
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
    with open(f'output/{log_file_name}', 'a+') as f:
        model.train()
        # remember to exit the train loop at end of the epoch
        for batch_idx, (data, target) in enumerate(train_loader):
            # Your code goes here!
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            model_out = model(data)
            loss = criterion(model_out, target)
            loss.backward()
            optimizer.step()

            elapsed_time = time.time() - start_time
            f.write(f"{epoch},{batch_idx+1},{elapsed_time}\n")
            start_time = time.time()

            if batch_idx % 20 == 0:
                log_iter_elapsed_time = time.time() - log_iter_start_time
                print(batch_idx, "loss: ", loss.item(), "Elapsed Time: ", log_iter_elapsed_time)
            log_iter_start_time = time.time()
        
        return None

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

    global log_file_name
    log_file_name = f"timelog.csv"
    with open(f'output/{log_file_name}', 'w+') as f:
        f.write("Epoch, Iteration, Elpased_time\n")
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
    train_loader = torch.utils.data.DataLoader(training_set,
                                                    num_workers=2,
                                                    batch_size=batch_size,
                                                    sampler=None,
                                                    shuffle=True,
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
    main()
