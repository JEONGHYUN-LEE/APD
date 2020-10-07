import torch
import torch.nn as nn
from transfer_resnet import get_alive_filters
import numpy as np


def element_pruning(layer, sparsity):
    weight = layer.weight.detach()
    abs_weight = torch.abs(weight)
    sorted_weight = torch.sort(abs_weight.reshape(-1))[0]
    threshold = sorted_weight[int(len(sorted_weight) * sparsity)]
    # print(len(sorted_weight),int(len(sorted_weight) * sparsity))
    with torch.no_grad():
        layer.weight[abs_weight < threshold] = 0.0


def get_arg_filtered(layer, sparsity):
    """Get filter index which has to be pruned(L2 norm based)"""
    # Setup
    weight = layer.weight.detach()
    num_filters = weight.shape[0]

    # Flattening each filters
    weight = weight.reshape(num_filters, -1)

    # Compute L2 norm of each filter
    filter_rank = torch.norm(weight, dim=1)

    # Sort filter index by Computed L2 norm(Ascending Order, Last biggest)
    arg_sort_rank = torch.argsort(filter_rank)

    # Compute threshold K
    arg_threshold = int(len(arg_sort_rank) * sparsity)

    # Return the index of K smallest L2 norm filter
    arg_filtered = arg_sort_rank[:arg_threshold]
    # print(len(arg_filtered))
    return arg_filtered


def conv_bn_fp(conv, bn, sparsity):
    """[Convolution - Batch Norm] routine pruning"""
    arg_filtered = get_arg_filtered(conv, sparsity)
    with torch.no_grad():
        # CONV
        conv.weight[arg_filtered] = 0.0

        # BN
        bn.weight[arg_filtered] = 0.0
        bn.bias[arg_filtered] = 0.0


def filter_pruning_status(model):
    total_alive_filters = 0
    total_filters = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            alive_filters = len(get_alive_filters(module))
            filters = module.weight.shape[0]

            total_alive_filters += alive_filters
            total_filters += filters
            print(name, ': ', alive_filters, '/', filters)
    return float(total_alive_filters)/float(total_filters)


def sparsity(model):
    sparsity = []
    for child in model.children():
        for layer in child.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                d2_weight = layer.weight.reshape(layer.weight.shape[0], -1)
                zero_filter = 0
                for weight in d2_weight:
                    if weight[weight != 0].sum() == 0:
                        zero_filter += 1
                total = layer.weight.numel()
                nonzero = layer.weight.nonzero().size(0)
                sparsity.append((total-nonzero)/total)
                print(layer, (total-nonzero)/total, zero_filter)
    return np.array(sparsity).mean()
