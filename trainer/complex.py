from attacks import *
from torch.nn import functional as F
import torch
from prun_resnet import *


def complex_train(params,
                  teacher,
                  loader,
                  model,
                  optimizer,
                  criterion,
                  device
                  ):
    model.train()

    # General training procedure
    corrects = 0
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()
        optimizer.zero_grad()

        # Adversarial Forward
        if params.adversarial_params:
            model.eval()
            # Use advertorch
            if params.adversarial_params.tool == 'advertorch':
                if params.adversarial_params.norm:
                    adversary = get_pgd_adversary(model,
                                                  params.adversarial_params.eps / 255.0,
                                                  params.adversarial_params.num_iter,
                                                  params.adversarial_params.lr / 255.0,
                                                  params.adversarial_params.rand_init,
                                                  params.general_params.random_seed)
                else:
                    adversary = get_pgd_adversary(model,
                                                  params.adversarial_params.eps,
                                                  params.adversarial_params.num_iter,
                                                  params.adversarial_params.lr,
                                                  params.adversarial_params.rand_init,
                                                  params.general_params.random_seed)
                    # print(params.adversarial_params.eps,
                    #                               params.adversarial_params.num_iter,
                    #                               params.adversarial_params.lr,)
                pert_x = adversary.perturb(x, y)
            # Use my
            else:
                pert_x = pgd_attack(model, x, y,
                                    params.adversarial_params.eps / 255.0,
                                    params.adversarial_params.num_iter,
                                    params.adversarial_params.lr / 255.0,
                                    params.adversarial_params.rand_init,
                                    params.general_params.random_seed)

            pert_x = pert_x.detach()

            optimizer.zero_grad()
            model.train()
            outputs = model(pert_x)

        # Original Forward
        else:
            outputs = model(x)

        predicts = torch.argmax(outputs, 1).detach()
        corrects += (predicts == y).sum()

        # Distillation Loss
        if teacher:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            lessons = teacher(x)

            t = params.distillation_params.temperature
            a = params.distillation_params.alpha

            dist_loss = F.kl_div((outputs / t).log_softmax(dim=1),
                                 (lessons / t).softmax(dim=1),
                                 reduction="batchmean")

            loss = (1.0 - a) * criterion(outputs, y) + a * (t ** 2) * dist_loss

        # None Distillation Loss
        else:
            loss = criterion(outputs, y)

        # Back Prop
        loss.backward()

        # Weight Update
        optimizer.step()

        # Projection
        if params.projection_params:
            if params.projection_params.type == 'ep':
                resnet_ep(model, params.projection_params.sparsity)
            elif params.projection_params.type == 'fp':
                resnet_fp(model, params.projection_params.sparsity)


def complex_test(params, loader, model, device):
    model.eval()
    corrects = 0

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).long()

        # Adversarial Forward
        if params.adversarial_params:
            if params.adversarial_params.tool == 'advertorch':
                if params.adversarial_params.norm:
                    adversary = get_pgd_adversary(model,
                                                  params.adversarial_params.eps / 255.0,
                                                  params.adversarial_params.num_iter,
                                                  params.adversarial_params.lr / 255.0,
                                                  params.adversarial_params.rand_init,
                                                  params.general_params.random_seed)
                else:
                    adversary = get_pgd_adversary(model,
                                                  params.adversarial_params.eps,
                                                  params.adversarial_params.num_iter,
                                                  params.adversarial_params.lr,
                                                  params.adversarial_params.rand_init,
                                                  params.general_params.random_seed)
                pert_x = adversary.perturb(x, y)
            else:
                pert_x = pgd_attack(model, x, y,
                                    params.adversarial_params.eps / 255.0,
                                    params.adversarial_params.num_iter,
                                    params.adversarial_params.lr / 255.0,
                                    params.adversarial_params.rand_init,
                                    params.general_params.random_seed)
            outputs = model(pert_x)

        # Original Forward
        else:
            outputs = model(x)

        predicts = torch.argmax(outputs, 1)
        corrects += (predicts == y).sum()

    return float(corrects) / float(len(loader.dataset))
