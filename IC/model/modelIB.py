import torch
import torch.nn.functional as F
from .resnet import *

class Robust_IB(nn.Module):
    def __init__(self, z_dim=256, num_classes=10, model_name='resnet34'):
        super(Robust_IB, self).__init__()
        self.S = resnet_IB(z_dim, model_name)
        self.T = resnet_IB(z_dim, model_name)
        
        self.decode = decoder(z_dim=z_dim, num_classes=num_classes)

    def forward(self, x, num_sample=1):
        (mu_S, std_S), encoding_S = self.S(x, num_sample)
        logit_S = self.decode(encoding_S)
        (mu_T, std_T), encoding_T = self.T(x, num_sample)
        logit_T = self.decode(encoding_T)

        return (mu_S, std_S), logit_S, (mu_T, std_T), logit_T

class IB(nn.Module):
    def __init__(self, z_dim=256, num_classes=10, model_name='resnet34'):
        super(IB, self).__init__()
        self.encode = resnet_IB(z_dim, model_name)    
        self.decode = decoder(z_dim=z_dim, num_classes=num_classes)

    def forward(self, x, num_sample=1):
        (mu, std), encoding = self.encode(x, num_sample)
        logit = self.decode(encoding)

        return (mu, std), logit

class decoder(nn.Module):
    def __init__(self, z_dim=256, hid_dim = 128, num_classes=10):
        super(decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, num_classes),
            nn.ReLU(inplace=True),
            nn.Linear(num_classes, num_classes)
        )

    def forward(self, x):
        logit = self.decode(x)

        return logit
    
class resnet_IB(nn.Module):
    def __init__(self, z_dim=256, model_name='resnet34'):
        super(resnet_IB, self).__init__()
        self.K = z_dim
        if model_name == 'resnet18':
            self.model = ResNet18(z_dim * 2)
        elif model_name == 'resnet34':
            self.model = ResNet34(z_dim * 2)
        elif model_name == 'resnet50':
            self.model = ResNet50(z_dim * 2)
        elif model_name == 'resnet101':
            self.model = ResNet101(z_dim * 2)
        elif model_name == 'resnet152':
            self.model = ResNet152(z_dim * 2)
        else:
            raise ValueError('Invalid model name')
        
    def forward(self, x, num_sample=1):
        statistics = self.model(x)
        mu = statistics[..., :self.K]
        std = F.softplus(statistics[..., self.K:]-5,beta=1)

        encoding = self.reparametrize_n(mu, std, num_sample)

        return (mu, std), encoding
    
    def reparametrize_n(self, mu, std, n=1):
        if n != 1 :
            mu = mu.unsqueeze(0).expand(n, *mu.shape)
            std = std.unsqueeze(0).expand(n, *std.shape)

        eps = torch.randn_like(std).to(mu.device)

        return mu + eps * std
    
    def reset(self):
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=2, num_classes=10):
        super(Discriminator, self).__init__()
        
        self.linear_x = nn.Linear(input_dim, hidden_dim)
        self.linear_y = nn.Linear(num_classes, hidden_dim)
        self.linear_sum = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x, y, xx, yy):
        h_x = F.relu(self.linear_x(x))
        h_y = F.relu(self.linear_y(y))
        h1 = self.linear_sum(torch.cat([h_x, h_y], dim=1))

        h_xx = F.relu(self.linear_x(xx))
        h_yy = F.relu(self.linear_y(yy))
        h2 = self.linear_sum(torch.cat([h_xx, h_yy], dim=1))

        h = torch.cat([h1, h2], dim=1)
        out = self.classifier(h)

        return out