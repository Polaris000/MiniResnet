import torch

from model import (
    MiniResNet,
    get_optimizers,
    MiniResNet_SingleChannel,
    get_optimizers_warmup,
)
from process import (
    load_data,
    augment_data_auto_config,
    augment_data_auto_config_normalize,
)

import torchsummary

import time

import pandas as pd
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import os


def train(
    model,
    train_loader,
    valid_loader,
    test_loader,
    epochs,
    criterion,
    optimizer,
    scheduler,
    early_stopper,
    device,
    writer,
    experiment,
):
    train_loss_history = []
    train_acc_history = []
    valid_loss_history = []
    valid_acc_history = []
    test_loss_history = []
    test_acc_history = []

    for epoch in range(epochs):
        start_time = time.time()

        train_loss, train_correct, train_total = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        valid_loss, valid_correct, valid_total = test(
            model, valid_loader, criterion, device
        )

        test_loss, test_correct, test_total = test(
            model, test_loader, criterion, device
        )

        train_loss = train_loss / len(train_loader)
        valid_loss = valid_loss / len(valid_loader)
        test_loss = test_loss / len(test_loader)

        train_acc = train_correct / train_total
        valid_acc = valid_correct / valid_total
        test_acc = test_correct / test_total

        train_loss_history += [train_loss]
        valid_loss_history += [valid_loss]
        test_loss_history += [test_loss]

        train_acc_history.append(train_acc)
        valid_acc_history.append(valid_acc)
        test_acc_history.append(test_acc)

        end_time = time.time()

        print(
            f"Epoch: {epoch + 1}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}, Train Acc: {train_acc:.3f}, Valid Acc: {valid_acc:.3f}, Test Acc: {test_acc:.3f}, Time: {(end_time - start_time) / 60:.2f} mins"
        )
        scheduler.step()

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/valid", valid_acc, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        if (epoch % 2 == 0) or early_stopper.early_stop(test_loss):
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": test_loss,
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")

            torch.save(state, f"./checkpoint/ckpt_{experiment}.pth")


def main(experiment, augment_config, optimizer_config):
    INPUT_DIM = (3, 32, 32)
    EPOCHS = 40
    DEVICE = "mps"

    train_loader, val_loader, test_loader, _, testloader_labeled = load_data(
        INPUT_DIM, augment_config
    )

    model = MiniResNet(num_blocks=[2, 1, 1, 1])
    model = model.to("cpu")
    torchsummary.summary(model, INPUT_DIM, device="cpu")

    model = model.to(DEVICE)

    assert (
        round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
        <= 5
    ), "Model Size excedes limit."

    criterion, optimizer, scheduler, early_stopper = optimizer_config(model)

    writer = SummaryWriter(f"runs/{experiment}")

    train(
        model,
        train_loader,
        val_loader,
        testloader_labeled,
        EPOCHS,
        criterion,
        optimizer,
        scheduler,
        early_stopper,
        DEVICE,
        writer,
        experiment,
    )

    writer.close()

    valid_loss, valid_correct, valid_total = test(model, val_loader, criterion, DEVICE)

    valid_acc = valid_correct / valid_total
    print(f"Valid Accuracy: {valid_acc}")

    test_loss, test_correct, test_total = test(
        model, testloader_labeled, criterion, DEVICE
    )

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc}")

    # results = infer(model, test_loader, criterion, DEVICE)
    # results.to_csv(f"../data/results/results_{experiment}.csv", index=False)


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
    EXPERIMENT = "auto_augment_cifar_10_40_epochs"
    augment_config = augment_data_auto_config
    optimizer_config = get_optimizers

    main(EXPERIMENT, augment_config, get_optimizers)

    EXPERIMENT = "auto_augment_cifar_10_40_epochs_normalize"
    augment_config = augment_data_auto_config_normalize
    optimizer_config = get_optimizers

    main(EXPERIMENT, augment_config, get_optimizers)

    EXPERIMENT = "auto_augment_cifar_10_40_epochs_normalize_cosine_warmup"
    augment_config = augment_data_auto_config_normalize
    optimizer_config = get_optimizers_warmup

    main(EXPERIMENT, augment_config, get_optimizers_warmup)

    EXPERIMENT = "auto_augment_cifar_10_40_epochs_cosine_warmup"
    augment_config = augment_data_auto_config
    optimizer_config = get_optimizers_warmup

    main(EXPERIMENT, augment_config, get_optimizers_warmup)
