# MiniResnet


### About


### Results

| Experiment | Test Acc | Code | Checkpoint |
|------------|---------|------|------------|
| Auto-augment | 0.771 | [`run.py`](miniresnet/run.py) | [checkpoint](miniresnet/checkpoint/ckpt_auto_augment_cifar_10_40_epochs.pth) |
| Auto-augment + Norm | 0.772 | [`run.py`](miniresnet/run.py) | [checkpoint](miniresnet/checkpoint/ckpt_auto_augment_cifar_10_40_epochs_normalize.pth) |
| Auto-augment + Warmup | 0.782  | [`run.py`](miniresnet/run.py) | [checkpoint](miniresnet/checkpoint/ckpt_auto_augment_cifar_10_40_epochs_cosine_warmup.pth) |
| Auto-augment + Warmup + Retrained | 0.787 | [`run.py`](miniresnet/run.py) | [checkpoint](miniresnet/checkpoint/auto_augment_cifar_10_40_epochs_cosine_warmup_resume.pth) |

---


### Experimental Setup




### Usage

#### Environment Setup
1. **Install Conda**: If you haven't already, install Conda by following the instructions on the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create a New Environment**: Open your terminal or command prompt and run the following command to create a new Conda environment. Replace `myenv` with your desired environment name.

    ```bash
    conda create --name myenv
    ```

3. **Activate the Environment**: Once the environment is created, activate it using the following command:

    ```bash
    conda activate myenv
    ```


4. **Install Dependencies**: Once you have your `requirements.txt` file ready, you can install all the dependencies listed in it using `pip`. Run the following command in your terminal:

    ```bash
    pip install -r requirements.txt
    ```


#### Running Notebook-based Experiments
Some of the experiments above were run via notebooks. Simply follow the general setup steps, then run the notebooks top-down.


#### Running Script-based Experiments
The experiments that are script-based are executed like this:
1. **Navigate to the `miniresnet` directory**
    ```
    cd miniresnet
    ```

2. **Setup the experiment by editing the code in run.py**  
    For example, an experiment can be setup like so:
    ```python
    ...
    if __name__ == "__main__":
        EXPERIMENT = "auto_augment_cifar_10_40_epochs"
        augment_config = augment_data_auto_config
        optimizer_config = get_optimizers

        main(EXPERIMENT, augment_config, get_optimizers)
    ```

3. **Model Saving**  
    Every few epochs, the model state is saved as a checkpoint with the name of the experiment.

4. **Training from Saved State**  
    In case training stops unexpectedly, or if you feel the need to train an already trained model for additional epochs, use the `resume` argument in the `main` function defined in [`run.py`](miniresnet/run.py).

    ```python
    if __name__ == "__main__":
        EXPERIMENT = "auto_augment_cifar_10_40_epochs_cosine_warmup_resume"
        augment_config = augment_data_auto_config
        optimizer_config = get_optimizers_warmup

        main(EXPERIMENT, augment_config, get_optimizers_warmup, resume=True)
    ```

5. **Model Loading**  
    To load a saved checkpoint and test it, use the `main_test` function in [`run.py`](miniresnet/run.py).

    ```python
    if __name__ == "__main__":
        EXPERIMENT = "auto_augment_cifar_10_40_epochs"
        augment_config = augment_data_auto_config
        optimizer_config = get_optimizers

        main_test(
            EXPERIMENT,
            augment_config,
            optimizer_config,
            resume=True
        )
    ```

    


---
### Files