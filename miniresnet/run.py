import torch

from model import MiniResNet, get_optimizers, MiniResNet_SingleChannel
from process import load_data

import torchsummary
import warnings

import pandas as pd
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


import os


def train(
    model,
    train_loader,
    test_loader,
    epochs,
    criterion,
    optimizer,
    scheduler,
    early_stopper,
    device,
    writer,
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
            model, test_loader, criterion, device
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
            f"Epoch {epoch + 1}, Train loss {train_loss:.3f}, Test loss {test_loss:.3f}, Train Accuracy: {train_acc:.3f}, Test Accuracy: {test_acc:.3f}"
        )
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        if (epoch % 10 == 0) or early_stopper.early_stop(test_loss):
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": test_loss,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")

            torch.save(state, "./checkpoint/ckpt_grayscale.pth")


def main():
    INPUT_DIM = (1, 32, 32)
    EPOCHS = 20
    DEVICE = "mps"

    train_loader, val_loader, test_loader, _ = load_data(INPUT_DIM)

    model = MiniResNet_SingleChannel(num_blocks=[2, 1, 1, 1])
    model = model.to(DEVICE)

    if DEVICE == "mps":
        warnings.warn("MPS not supported by torchsummary.")

    else:
        print(torchsummary.summary(model, INPUT_DIM, device=DEVICE))

    assert (
        round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
        <= 5
    ), "Model Size excedes limit."

    criterion, optimizer, scheduler, early_stopper = get_optimizers(model)

    writer = SummaryWriter("runs/example", comment="MiniResNet_SingleChannel")

    train(
        model,
        train_loader,
        val_loader,
        EPOCHS,
        criterion,
        optimizer,
        scheduler,
        early_stopper,
        DEVICE,
        writer,
    )

    writer.close()

    valid_loss, valid_correct, valid_total = test(model, val_loader, criterion, DEVICE)

    valid_acc = valid_correct / valid_total
    print(f"Valid Accuracy: {valid_acc}")

    test_loss, test_correct, test_total = test(model, test_loader, criterion, DEVICE)

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc}")

    results = infer(model, test_loader, criterion, DEVICE)
    results.to_csv("../data/results/results_grayscale.csv", index=False)


def main_test():
    DEVICE = "mps"

    model = load_model(DEVICE)
    _, _, test_loader, _ = load_data((3, 32, 32))
    criterion, _, _, _ = get_optimizers(model)

    results = infer(model, test_loader, criterion, DEVICE)

    results.to_csv("results.csv", index=False)


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


def test(model, test_loader, criterion, device):
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


def infer(model, test_loader, criterion, device):
    model.eval()

    results = []

    for _, (id_, image) in tqdm(enumerate(test_loader)):
        image = image.to(device)
        output = model(image)
        _, predicted = output.max(1)
        results.append({"ID": id_.item(), "Labels": predicted.item()})

    return pd.DataFrame(results)


def load_model(path="./checkpoint/ckpt.pth", DEVICE="mps"):
    model = MiniResNet(num_blocks=[1, 1, 1, 1])
    model.load_state_dict(torch.load(path)["state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model


if __name__ == "__main__":
    main()
