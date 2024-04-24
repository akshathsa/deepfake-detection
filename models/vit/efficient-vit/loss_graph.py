import os
from glob import glob
import re
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = '/global/cfs/projectdirs/m3641/Akaash/deepfake-detection/'
MODELS_PATH = os.path.join(BASE_DIR, "models")

train = None
val = None
for path in glob(os.path.join(MODELS_PATH, "output/losses", "*")):
    with open(path, "r") as f:
        if "train" in path:
            train = f.read()
        else:
            val = f.read()

train_losses = train.split("train loss: ")
val_losses = val.split("val loss: ")

for i in range(len(train_losses)):
    train_losses[i] = train_losses[i][:train_losses[i].find("Epoch")]

for i in range(len(val_losses)):
    val_losses[i] = val_losses[i][:val_losses[i].find("Epoch")]

# 200 epochs
train_losses = [float(t) for t in train_losses if len(t) != 0][:200]
val_losses = [float(v) for v in val_losses if len(v) != 0][:200]

x = np.arange(200)
plt.plot(x, train_losses, label="Train Losses")
plt.plot(x, val_losses, label="Validation Losses")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("../../output/losses.png")
