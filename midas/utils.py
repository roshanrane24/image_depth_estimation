import torch
import torch.nn as nn

def _make_encoder(features, use_pretrained):
    pretrained = _make_pretrained_resnext101_wsl(use_pretrained)
    scratch = _make_scratch([256, 512, 1024, 2048], features)

    return pretrained, scratch

def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
                        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
                        )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained

def _make_pretrained_resnext101_wsl(use_pretrained):
    resnet = torch.hub.load("facebookresearch/WSL-Images", "resnext101_32x8d_wsl")
    return _make_resnet_backbone(resnet)


def _make_scratch(in_shape, out_shape):
    scratch = nn.Module()

    scratch.layer1_rn = nn.Conv2d(
                        in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False
                        )
    scratch.layer2_rn = nn.Conv2d(
                        in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False
                        )
    scratch.layer3_rn = nn.Conv2d(
                        in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False
                        )
    scratch.layer4_rn = nn.Conv2d(
                        in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False
                        )
    return scratch
