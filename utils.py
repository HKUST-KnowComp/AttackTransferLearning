import numpy as np
import torch
import torch.nn as nn
from collections import deque


class EarlyStopping(object):
    """EarlyStopping handler can be used to stop the training if no improvement after a given number of events
    Args:
        patience (int):
            Number of events to wait if no improvement and then stop the training
    """
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.meter = deque(maxlen=patience)

    def is_stop_training(self, score):
        stop_sign = False
        self.meter.append(score)
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                stop_sign = True
        # approximately equal
        elif np.abs(score - self.best_score) < 1e-9:
            if len(self.meter) == self.patience and np.abs(np.mean(self.meter) - score) < 1e-7:
                stop_sign = True
            else:
                self.best_score = score
                self.counter = 0
        else:
            self.best_score = score
            self.counter = 0
        return stop_sign
