"""Loss functions, optimizers, and schedulers."""
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np


class LinearLRWithWarmup(LambdaLR):
    """
    Linear learning rate scheduler with linear warmup.
    Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/optimization.py#L75
    
    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to LambdaLR
    warmup_steps: int
    total_steps: int
    """
    def __init__(self, *args, warmup_steps, total_steps, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(*args, **kwargs, lr_lambda=self.lr_lambda)
            
    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        return max(
            0.0, 
            float(self.total_steps - current_step) / float(max(1, self.total_steps - self.warmup_steps))
        )

class InverseSqrtLRWithWarmup(LambdaLR):
    """
    Linear learning rate scheduler with linear warmup.
    Adapted from https://github.com/huggingface/transformers/blob/v4.23.0/src/transformers/optimization.py#L75
    
    Parameters
    ----------
    *args, **kwargs: additionnal arguments are passed to LambdaLR
    warmup_steps: int
    total_steps: int
    """
    def __init__(self, *args, warmup_steps, total_steps, update_factor, **kwargs):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.update_factor = update_factor if update_factor is not None else (warmup_steps if warmup_steps > 10 else total_steps / 20)
        super().__init__(*args, **kwargs, lr_lambda=self.lr_lambda)
            
    def lr_lambda(self, current_step: int):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        # return 1 / np.sqrt(float(current_step - self.warmup_steps) / float(self.total_steps - self.warmup_steps) * self.update_factor ** 2 + 1)
        return 1 / np.sqrt(float(current_step - self.warmup_steps) / float(self.update_factor) + 1)

class LabelSmoothingLoss(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, logprobs, target):
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class MultiNLLLoss(nn.Module):
    """
    - sum_w log(p_w | X) 
    """

    def __init__(self, label_smoothing=0.0):
        super(MultiNLLLoss, self).__init__()
        self.ls = label_smoothing

    def forward(self, logprobs, target_mask):
        # logprobs : B x V
        # target_mask : B x V
        if self.ls > 0:
            smooth_loss = -logprobs.mean(-1).mean()
        else:
            smooth_loss = 0
        masked_logprobs = logprobs.masked_fill(~target_mask, 0)
        no_empty = target_mask.sum(-1) > 0
        loss = -(masked_logprobs[no_empty].sum(-1) / target_mask[no_empty].sum(-1)).mean()
        return self.ls * smooth_loss + (1.0 - self.ls) * loss
        # else:
        #     if self.reduction == "mean":
        #         return -logprobs[target_mask].mean()
        #     else:
        #         return -logprobs[target_mask].sum()

class MSELevenshteinLoss_old(nn.Module):
    def __init__(self, alpha=0.6, beta=-1.0):
        super(MSELevenshteinLoss, self).__init__()
        self.alpha = nn.parameter.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.parameter.Parameter(torch.tensor(beta, dtype=torch.float32))
        self.alpha_norm = nn.Tanh() # between -1 and 1
        self.beta_norm = nn.Sigmoid() # between 0 and 1
        self.mse = nn.MSELoss()

    def get_normalized_alpha(self):
        with torch.no_grad():
            return self.alpha_norm(self.alpha)

    def get_normalized_beta(self):
        with torch.no_grad():
            return self.beta_norm(self.beta)

    def get_pseudo_lev(self, similarities):
        thresh_a = self.alpha_norm(self.alpha)
        thresh_b = self.beta_norm(self.beta)
        msk = similarities < thresh_a
        left = (similarities[msk] + 1) * thresh_b / (1 + thresh_a)
        right = ((thresh_b - 1) * similarities[~msk] + thresh_a - thresh_b) / (thresh_a - 1)
        pseudo_lev = torch.empty_like(similarities)
        pseudo_lev[msk] = left
        pseudo_lev[~msk] = right
        return pseudo_lev

    def forward(self, similarities, levs):
        return self.mse(self.get_pseudo_lev(similarities).view(-1), levs.view(-1))

class MSELevenshteinLoss(nn.Module):
    def __init__(self, alpha=4.0, beta=-3.0, loss_type="mse"):
        super(MSELevenshteinLoss, self).__init__()
        self.alpha = nn.parameter.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.beta = nn.parameter.Parameter(torch.tensor(beta, dtype=torch.float32))
        if loss_type == "mae":
            self._loss = nn.L1Loss()
        else:
            self._loss = nn.MSELoss()

    def get_normalized_alpha(self):
        return self.alpha.detach()

    def get_normalized_beta(self):
        return self.beta.detach()

    def get_pseudo_lev(self, similarities):
        similarities = similarities.clamp(-0.99, 0.99)
        x = (self.alpha * torch.log((1 - similarities) / (1 + similarities)) / 2 - self.beta).clamp(-80, 80)

        return 1 / (1 + torch.exp(x))

    def forward(self, similarities, levs):
        return self._loss(self.get_pseudo_lev(similarities).view(-1), levs.view(-1))


class AdaptiveMarginRankLoss(nn.Module):
    def __init__(self, sigma=1.0):
        super(AdaptiveMarginRankLoss, self).__init__()
        self.sigma = sigma

    def forward(self, similarities, levs):
        levs = levs.view(similarities.shape)
        rank = torch.argsort(levs, -1)
        B, N = levs.shape
        # print("sims", similarities.shape)
        # print("levs", levs.shape)
        # print("rank", rank.shape, rank.min().item(), rank.max().item())
        C = (
            torch.abs(
                levs.gather(1, rank).view(B, N, 1) - levs.gather(1, rank).view(B, 1, N)
            ) * self.sigma
            + similarities.gather(1, rank).view(B, N, 1) - similarities.gather(1, rank).view(B, 1, N)
        ).triu(diagonal=1).clamp(min=0)
        return C.mean()