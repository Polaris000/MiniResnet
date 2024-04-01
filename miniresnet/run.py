import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from model import MiniResNet, get_optimizers
from process import load_data

import os


def train(model, train_loader, test_loader, epochs, criterion, optimizer, scheduler):
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in epochs:
        train_loss, train_correct, train_total = train_epoch(model, criterion, optimizer)
        test_loss, test_correct, test_total = test(model, criterion, optimizer)

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total

        train_loss_history += [train_loss]
        test_loss_history += [test_loss]

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(f'Epoch {epoch}, Train loss {train_loss}, Test loss {test_loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}')
        scheduler.step()

    if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')


def main():
    train_loader, test_loader = load_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MiniResNet()
    model = model.to(device)

    criterion, optimizer, scheduler = get_optimizers()
    train(model, train_loader, test_loader, epochs, criterion, optimizer, scheduler)

    test_loss, test_correct, test_total = test(model, test_loader, epochs, criterion, optimizer)

    test_acc = test_correct / test_total
    print("Test Accuracy: {test_acc}")



def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return train_loss, correct, total


def test(model, test_loader, criterion, optimizer):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss, correct, total


def main():
    train_loader, test_loader = load_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MiniResNet()
    model = model.to(device)

    criterion, optimizer, scheduler = get_optimizers()
    train(model, train_loader, test_loader, epochs, criterion, optimizer, scheduler)

    test_loss, test_correct, test_total = test(model, test_loader, epochs, criterion, optimizer)

    test_acc = test_correct / test_total
    print("Test Accuracy: {test_acc}")


if __name__ == "__main__":
    main()