class EarlyStopping():
    def __init__(self, tolerance=3):

        self.tolerance = tolerance
        self.last_val_loss = 0
        self.last_train_loss = 0
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if train_loss < self.last_train_loss:
            if validation_loss > self.last_val_loss:
                self.counter +=1
                if self.counter >= self.tolerance:  
                    self.early_stop = True

        self.last_train_loss = train_loss
        self.last_val_loss = validation_loss