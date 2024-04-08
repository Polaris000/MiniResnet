import os

# Training parameters.
EPOCHS = 50
# Data root.
DATA_ROOT = os.path.dirname(__file__) + "/../data"
# Number of parallel processes for data fetching.
NUM_WORKERS = 2
# Ratio of split to use for validation.
VALID_SPLIT = 0.1
# Image to resize to in tranforms.
IMAGE_SIZE = 224
# For ASHA scheduler in Ray Tune.
MAX_NUM_EPOCHS = 50
GRACE_PERIOD = 1
# For search run (Ray Tune settings).
CPU = 1
GPU = 1
# Number of random search experiments to run.
NUM_SAMPLES = 20
