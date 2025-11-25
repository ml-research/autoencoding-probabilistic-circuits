#!/usr/bin/env python3


import numpy as np

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience, delta_rel=0.01):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = np.inf
        self.delta_rel = delta_rel

    def check_stop(self, val_loss) -> bool:

        # If the validation loss is at least 1% lower than the best validation loss
        if val_loss > (self.best_val_loss - self.delta_rel * self.best_val_loss):
            # Increase counter
            self.counter += 1

            # Return true (i.e., stop) if counter >= patience
            return self.counter >= self.patience
        else:
            # Save new best loss
            self.best_val_loss = val_loss

            # Reset counter
            self.counter = 0

            # Return false (i.e., don't stop)
            return False
