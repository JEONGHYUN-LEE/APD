from prun import *
import torch.nn as nn


def residual_block_fp(block, sparsity):
    """Residual Block Pruning"""
    # First CONV-BN
    conv_bn_fp(block.conv1, block.bn1, sparsity)

    # Second CONV-BN
    conv_bn_fp(block.conv2, block.bn2, sparsity)

    # Shortcut CONV-BN
    if len(block.shortcut):
        conv_bn_fp(block.shortcut[0], block.shortcut[1], sparsity)


def resnet_ep(model, sparsity):
    """ResNet whole network pruning"""
    for child in model.children():
        for layer in child.modules():
            if isinstance(layer, nn.Conv2d):
                element_pruning(layer, sparsity)


def resnet_fp(model, sparsity):
    """ResNet whole network pruning"""
    # First Conv Layer
    conv_bn_fp(model.conv1, model.bn1, sparsity)

    # First Blocks
    for block in model.layer1:
        residual_block_fp(block, sparsity)

    # Second Blocks
    for block in model.layer2:
        residual_block_fp(block, sparsity)

    # Third Blocks
    for block in model.layer3:
        residual_block_fp(block, sparsity)

    # Forth Blocks
    for block in model.layer4:
        residual_block_fp(block, sparsity)
