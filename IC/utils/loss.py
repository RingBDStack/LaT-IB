import torch
import torch.nn as nn
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
    
def cross_entropy_soft(outputs, targets):
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- targets.detach() * logsoftmax(outputs), 1))

def ConCE(prob1, prob2, y, thod=0.90, n_type='sym'):
    assert len(prob1) == len(prob2)
    assert len(prob1) == len(y)
    
    a = F.cross_entropy(prob1.clone().detach(), y, reduction='none')
    b = F.cross_entropy(prob2.clone().detach(), y, reduction='none')
    
    pred1 = torch.argmax(prob1.clone().detach(), dim=1)
    pred2 = torch.argmax(prob2.clone().detach(), dim=1)
    mask_a = (a <= b)
    mask_b = (a > b) | (pred1 == pred2)
    
    w = (pred1 == pred2).sum()/len(pred1 == pred2)
    if w.item() > thod: 
        mask_a = (a <= b) | (pred1 == pred2)
        mask_b = (a > b)
        
    y1 = y.clone().detach()
    y2 = y.clone().detach()
    y1[mask_a] = pred1[mask_a]
    y2[mask_b] = pred2[mask_b]
    
    loss1 = F.cross_entropy(prob1, y1, reduction='mean')
    if n_type == 'asym':
        loss2 = F.cross_entropy(prob2, y, reduction='mean')
    else:
        loss2 = F.cross_entropy(prob2, y2, reduction='mean')
    final_loss = loss1 + loss2
    
    return final_loss

def kl_gaussian_diag(mu_P, std_P, mu_Q, std_Q, reduction='mean'):
    """
    Compute the KL divergence between two d-dimensional diagonal Gaussian distributions:
    N(mu_P, diag(std_P^2)) and N(mu_Q, diag(std_Q^2))
    
    Add epsilon to avoid numerical instability
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

    # 计算 JS 散度
    kl_P_M = kl_gaussian_diag(mu_P, std_P, mu_M, std_M, reduction=reduction)
    kl_Q_M = kl_gaussian_diag(mu_Q, std_Q, mu_M, std_M, reduction=reduction)
    
    return 0.5 * (kl_P_M + kl_Q_M)


def discriminator_loss(output, joint=False):
    if joint:
        target = torch.ones_like(output)
    elif not joint:
        target = torch.zeros_like(output)
    return F.cross_entropy(output, target)


def debias_pl(logit,bias,tau=0.8):
    bias = bias.detach().clone()
    debiased_prob = F.softmax(logit - tau*torch.log(bias), dim=1)
    return debiased_prob


def debias_output(logit, bias, tau=0.8):
    bias = bias.detach().clone()
    debiased_opt = logit + tau*torch.log(bias)
    return debiased_opt

def bias_initial(num_class=10):
    bias = (torch.ones(num_class, dtype=torch.float)/num_class).cuda()
    return bias

def bias_update(input, bias, momentum=0.9999, bias_mask=None):
    if input.numel() == 0:
        return bias
    if bias_mask is not None:
        input_mean = input.detach()*bias_mask.detach().unsqueeze(dim=-1)
    else:
        input_mean = input.detach().mean(dim=0)
    bias = momentum * bias + (1 - momentum) * input_mean
    return bias

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length\

class GCELoss(torch.nn.Module):
    """
    GCE: Generalized Cross Entropy
    2018 NeurIPS | Generalized cross entropy loss for training deep neural networks with noisy labels
    Ref: https://github.com/AlanChou/Truncated-Loss/blob/master/TruncatedLoss.py
    """
    def __init__(self, q=0.9):
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