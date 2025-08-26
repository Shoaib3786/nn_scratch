
import torch

class DataGen():

    def __init__(self):
        torch.manual_seed(42)   # keep the static starting point for generating random numbers
        n_samples = 600
        self.x1 = torch.randn(n_samples, 1) * 0.5 + torch.randint(0, 3, (n_samples,1)).float()
        self.x2 = torch.randn(n_samples, 1) * 0.5 + torch.randint(0, 3, (n_samples,1)).float()
        # Define target with some nonlinear relationship
        self.y = ((self.x1**2 + self.x2) > 2.5).float()
        self.y += 0.1*torch.randn_like(self.y)   # add label noise
        self.y = (self.y > 0.5).float()
        self.X = torch.cat([self.x1, self.x2], dim=1)
    
    # Split train / test
    def dataSplit(self):
        train_len = int(0.8 * len(self.X))
        self.X_train, self.y_train = self.X[:train_len], self.y[:train_len]
        self.X_test, self.y_test   = self.X[train_len:], self.y[train_len:]
        # print("Class balance:", self.y.mean().item())
        # print("X_train shape:", self.X_train.shape, "y_train shape:", self.y_train.shape)
        return self.X_train, self.y_train, self.X_test, self.y_test