from advertorch.attacks import LinfPGDAttack
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random


def get_pgd_adversary(model, eps, num_iter, lr, rand_init, seed):

    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    adversary = LinfPGDAttack(model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=eps, nb_iter=num_iter,
                              rand_init=rand_init, eps_iter=lr, clip_min=0.0, clip_max=1.0)
    return adversary


def pgd_attack(model, input, target, eps, num_iter, lr, rand_init, seed):
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    x = input.detach()

    if rand_init:
        x += torch.zeros_like(x).uniform_(-eps, eps)

    for i in range(num_iter):
        x.requires_grad_()
        with torch.enable_grad():
            logits = model(x)
            loss = F.cross_entropy(logits, target, size_average=False)
        grad = torch.autograd.grad(loss, [x])[0]
        x = x.detach() + lr * torch.sign(grad.detach())
        x = torch.min(torch.max(x, input - eps), input + eps)

        x = torch.clamp(x, 0, 1)

    return x
