# https://debuggercafe.com/hyperparameter-tuning-with-pytorch-and-ray-tune/


from model import CustomNet
from process import load_data
from config import (
    MAX_NUM_EPOCHS,
    GRACE_PERIOD,
    EPOCHS,
    CPU,
    GPU,
    NUM_SAMPLES,
    DATA_ROOT_DIR,
    NUM_WORKERS,
    IMAGE_SIZE,
    VALID_SPLIT,
)
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os

from run import train, test


def train_and_validate(config):
    # Get the data loaders.
    train_loader, valid_loader, dataset_classes = load_data(
        IMAGE_SIZE, DATA_ROOT_DIR, VALID_SPLIT, config["batch_size"], NUM_WORKERS
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize the model.
    model = CustomNet(config["first_conv_out"], config["first_fc_out"]).to(device)
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    # Start the training.
    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion, device
        )
        valid_epoch_loss, valid_epoch_acc = test(model, valid_loader, criterion, device)

        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        print("-" * 50)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(loss=valid_epoch_loss, accuracy=valid_epoch_acc)


def run_search():
    # Define the parameter search configuration.
    config = {
        "first_conv_out": tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        "first_fc_out": tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
    }
    # Schduler to stop bad performing trails.
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=MAX_NUM_EPOCHS,
        grace_period=GRACE_PERIOD,
        reduction_factor=2,
    )
    # Reporter to show on command line/output window
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
    # Start run/search
    result = tune.run(
        train_and_validate,
        resources_per_trial={"cpu": CPU, "gpu": GPU},
        config=config,
        num_samples=NUM_SAMPLES,
        scheduler=scheduler,
        local_dir="../outputs/raytune_result",
        keep_checkpoints_num=1,
        checkpoint_score_attr="min-validation_loss",
        progress_reporter=reporter,
    )
    # Extract the best trial run from the search.
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation acc: {best_trial.last_result['accuracy']}")


if __name__ == "__main__":
    run_search()
