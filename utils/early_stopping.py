class EarlyStopping:
    """A helper class tracking the progress on the validation set
    If no progress is made for a pre-set time, it indicates that the training process should be stopped
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'Stopper no improvement: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
