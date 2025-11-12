import torch.nn.functional as F
import torch
from scipy.stats import beta
import numpy as np

class Mixup:
    def __init__(self, alpha=5):
        super().__init__()
        self.alpha = alpha
        self.index = None
        self.lam = None

    def loss(self, y_pred, y):
        mixed_target = self.lam * y + (1 - self.lam) * y[self.index]
        return F.cross_entropy(y_pred, mixed_target)
    
    def __call__(self, x):
        self.lam = torch.tensor(beta.rvs(self.alpha, self.alpha), device=x.device)
        self.lam = torch.maximum(self.lam, 1 - self.lam)
        # lam = np.random.beta(self.alpha, self.alpha)
        # self.lam = max(lam, 1 - lam)
        self.index = torch.randperm(x.size(0)).to(x.device)
        mixed_x = self.lam * x + (1 - self.lam) * x[self.index]
        return mixed_x
