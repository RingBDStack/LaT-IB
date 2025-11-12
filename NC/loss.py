import torch
import torch.nn.functional as F

def Shannon_Entropy(outputs, reduction='mean'):
    outputs = outputs.clamp(min=1e-12)
    probs = torch.softmax(outputs, dim=1)
    if reduction == 'mean':
        return torch.mean(-torch.sum(probs.log() * probs, dim=1))
    elif reduction == 'none':
        return -torch.sum(probs.log() * probs, dim=1)
    else:
        raise ValueError('Invalid reduction type')

def ConCE(prob1, prob2, y, beta=5.0, thod=0.9, noise_mode='uniform'):
    assert len(prob1) == len(prob2)
    assert len(prob1) == len(y)
    conf1, _ = torch.max(F.softmax(prob1.clone().detach(), dim=1), dim=1)
    pred1 = torch.argmax(prob1.clone().detach(), dim=1)
    y[(conf1 > thod)] = pred1[(conf1 > thod)]

    a = F.cross_entropy(prob1, y, reduction='none')
    b = F.cross_entropy(prob2, y, reduction='none')

    # 1/β ·log(exp(−β ·a)+exp(−β ·b))
    final_loss = torch.logsumexp(torch.stack([-beta * a, -beta * b]), dim=0) / -beta
    final_loss = final_loss.mean()
    return final_loss

def kl_gaussian_diag(mu_P, std_P, mu_Q, std_Q, reduction='mean'):
    """
    Compute the KL divergence between two d-dimensional diagonal Gaussian distributions:
    N(mu_P, diag(std_P^2)) and N(mu_Q, diag(std_Q^2))

    Add epsilon to avoid numerical instability.
    """
    epsilon = 1e-8  
    
    std_P = std_P.clamp(min=epsilon)
    std_Q = std_Q.clamp(min=epsilon)
    
    term1 = (std_P.pow(2) + epsilon) / (std_Q.pow(2) + epsilon)
    term2 = ((mu_Q - mu_P).pow(2)) / (std_Q.pow(2) + epsilon)
    term3 = torch.log((std_Q.pow(2) + epsilon) / (std_P.pow(2) + epsilon))
    if reduction == 'mean':
        kl_div = 0.5 * (term1 + term2 - 1 + term3).sum(1).mean()
    elif reduction == 'none':
        kl_div = 0.5 * (term1 + term2 - 1 + term3).sum(1)
    return kl_div

def js_gaussian_diag(mu_P, std_P, mu_Q, std_Q, reduction='mean'):
    """
    Compute the Jensen-Shannon (JS) divergence between two d-dimensional diagonal Gaussian distributions:
    N(mu_P, diag(std_P^2)) and N(mu_Q, diag(std_Q^2))
    """
    epsilon = 1e-8

    std_P = std_P.clamp(min=epsilon)
    std_Q = std_Q.clamp(min=epsilon)
    
    mu_M = 0.5 * (mu_P + mu_Q)
    
    std_M = torch.sqrt(0.5 * (std_P.pow(2) + std_Q.pow(2)) + 0.25 * (mu_P - mu_Q).pow(2) + epsilon)

    kl_P_M = kl_gaussian_diag(mu_P, std_P, mu_M, std_M, reduction=reduction)
    kl_Q_M = kl_gaussian_diag(mu_Q, std_Q, mu_M, std_M, reduction=reduction)
    
    return 0.5 * (kl_P_M + kl_Q_M)

def discriminator_loss(output, joint=False):
    if joint:
        target = torch.ones_like(output)
    elif not joint:
        target = torch.zeros_like(output)
    return F.cross_entropy(output, target)

class GCELoss(torch.nn.Module):
    """
    GCE: Generalized Cross Entropy
    2018 NeurIPS | Generalized cross entropy loss for training deep neural networks with noisy labels
    Ref: https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
    """
    def __init__(self, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q

    def forward(self, preds, labels):
        pred = F.softmax(preds, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        Yg = torch.gather(pred, 1, torch.unsqueeze(labels, 1).long())
        Lq = ((1-(Yg**self.q))/self.q)
        return torch.mean(Lq)

class SCELoss(torch.nn.Module):
    """
    SCE: Symmetric Cross Entropy
    2019 ICCV | Symmetric cross entropy for robust learning with noisy labels
    Ref: https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
    """
    def __init__(self, alpha, beta, num_classes=2):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        # CCE
        ce = self.cross_entropy(preds, labels.long())

        # RCE
        pred = F.softmax(preds, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels.long(), self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss