import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl


def train_val_test(gvf, cfg):
    # Train GVF on offline training set
    train_loss = gvf.train()

    # Save training loss, gvf, and optimizer
    np.save(cfg.exp_path + "/train_loss.npy", np.array(train_loss))
    gvf.save()

    # Evaluate GVF on validation set
    validation_loss = gvf.validation_loss()
    print("Validation Loss:", validation_loss)

    # Evaluate GVF on test set
    test_loss = gvf.test_loss()
    print("Test Loss:", test_loss)

    # Plot training loss
    fig, ax = plt.subplots()
    ax.plot(range(len(train_loss)), train_loss)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Batch Loss")
    fig.savefig(cfg.exp_path + "/gvf_training_loss.png")
    plt.close(fig)