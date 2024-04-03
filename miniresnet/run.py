import torch

from model import MiniResNet, get_optimizers
from process import load_data

import torchsummary

import os


def train(
    model, train_loader, test_loader, epochs, criterion, optimizer, scheduler, device
):
    train_loss_history = []
    train_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(epochs):
        train_loss, train_correct, train_total = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_correct, test_total = test(
            model, test_loader, criterion, optimizer, device
        )

        train_loss = train_loss / len(train_loader)
        test_loss = test_loss / len(test_loader)

        train_acc = train_correct / train_total
        test_acc = test_correct / test_total

        train_loss_history += [train_loss]
        test_loss_history += [test_loss]

        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)

        print(
            f"Epoch {epoch}, Train loss {train_loss}, Test loss {test_loss}, Train Accuracy: {train_acc}, Test Accuracy: {test_acc}"
        )
        scheduler.step()

        if epoch % 10 == 0:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
                torch.save(state, "./checkpoint/ckpt.pth")


def main():
    INPUT_DIM = (3, 32, 32)
    EPOCHS = 10
    DEVICE = "mps"

    train_loader, val_loader, test_loader, _ = load_data(INPUT_DIM)

    model = MiniResNet(num_blocks=[1, 1, 1, 1])
    model = model.to(DEVICE)

    if DEVICE == "mps":
        raise NotImplementedError("MPS not supported by torchsummary.")

    print(
        torchsummary.summary(model, INPUT_DIM),
        device=DEVICE,
    )

    assert (
        round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
        <= 5
    ), "Model Size excedes limit."

    criterion, optimizer, scheduler = get_optimizers(model)

    train(
        model,
        train_loader,
        test_loader,
        EPOCHS,
        criterion,
        optimizer,
        scheduler,
        DEVICE,
    )

    valid_loss, valid_correct, valid_total = test(
        model, val_loader, EPOCHS, criterion, optimizer, DEVICE
    )

    valid_acc = valid_correct / valid_total
    print(f"Valid Accuracy: {valid_acc}")

    test_loss, test_correct, test_total = test(
        model, test_loader, EPOCHS, criterion, optimizer, DEVICE
    )

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for _, (inputs, targets) in enumerate(train_loader):
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


def test(model, test_loader, criterion, optimizer, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss, correct, total


if __name__ == "__main__":
    main()
