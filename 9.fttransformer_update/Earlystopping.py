class EarlyStopping:
    def __init__(self, patience=10, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_state_dict = None
       

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state_dict = model.state_dict()
            self.counter = 0
         
            if self.verbose:
                print(f"Validation loss 개선: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss 개선되지 않음. patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print("★학습 종료★")
                self.early_stop = True
