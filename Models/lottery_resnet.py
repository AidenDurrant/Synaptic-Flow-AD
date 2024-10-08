# Based on code taken from https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
from Layers import layers_class
import torch

class Block(nn.Module):
    """A ResNet block."""

    def __init__(self, f_in: int, f_out: int, downsample=False):
        super(Block, self).__init__()

        stride = 2 if downsample else 1
        self.conv1 = layers_class.Conv2d(f_in, f_out, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layers_class.BatchNorm2d(f_out)
        self.conv2 = layers_class.Conv2d(f_out, f_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layers_class.BatchNorm2d(f_out)

        # No parameters for shortcut connections.
        if downsample or f_in != f_out:
            self.shortcut = nn.Sequential(
                layers_class.Conv2d(f_in, f_out, kernel_size=1, stride=2, bias=False),
                layers_class.BatchNorm2d(f_out)
            )
        else:
            self.shortcut = layers_class.Identity2d(f_in)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """A residual neural network as originally designed for CIFAR-10."""
    
    def __init__(self, plan, num_classes, dense_classifier):
        super(ResNet, self).__init__()

        # Initial convolution.
        current_filters = plan[0][0]
        self.conv = layers_class.Conv2d(3, current_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = layers_class.BatchNorm2d(current_filters)

        # The subsequent blocks of the ResNet.
        blocks = []
        for segment_index, (filters, num_blocks) in enumerate(plan):
            for block_index in range(num_blocks):
                downsample = segment_index > 0 and block_index == 0
                blocks.append(Block(current_filters, filters, downsample))
                current_filters = filters

        self.blocks = nn.Sequential(*blocks)

        self.fc = layers_class.Linear(plan[-1][0], num_classes)
        if dense_classifier:
            self.fc = nn.Linear(plan[-1][0], num_classes)

        self._initialize_weights()


    def forward(self, x):
        out = F.relu(self.bn(self.conv(x)))
        out = self.blocks(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers_class.Linear, nn.Linear, layers_class.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers_class.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def _plan(D, W):
    """The naming scheme for a ResNet is 'cifar_resnet_N[_W]'.

    The ResNet is structured as an initial convolutional layer followed by three "segments"
    and a linear output layer. Each segment consists of D blocks. Each block is two
    convolutional layers surrounded by a residual connection. Each layer in the first segment
    has W filters, each layer in the second segment has 32W filters, and each layer in the
    third segment has 64W filters.

    The name of a ResNet is 'cifar_resnet_N[_W]', where W is as described above.
    N is the total number of layers in the network: 2 + 6D.
    The default value of W is 16 if it isn't provided.

    For example, ResNet-20 has 20 layers. Exclusing the first convolutional layer and the final
    linear layer, there are 18 convolutional layers in the blocks. That means there are nine
    blocks, meaning there are three blocks per segment. Hence, D = 3.
    The name of the network would be 'cifar_resnet_20' or 'cifar_resnet_20_16'.
    """
    if (D - 2) % 3 != 0:
        raise ValueError('Invalid ResNet depth: {}'.format(D))
    D = (D - 2) // 6
    plan = [(W, D), (2*W, D), (4*W, D)]

    return plan

def _resnet(arch, plan, num_classes, dense_classifier, pretrained):
    model = ResNet(plan, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


# ResNet Models
def resnet20(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 16)
    return _resnet('resnet20', plan, num_classes, dense_classifier, pretrained)

def resnet32(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(32, 16)
    return _resnet('resnet32', plan, num_classes, dense_classifier, pretrained)

def resnet44(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(44, 16)
    return _resnet('resnet44', plan, num_classes, dense_classifier, pretrained)

def resnet56(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(56, 16)
    return _resnet('resnet56', plan, num_classes, dense_classifier, pretrained)

def resnet110(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(110, 16)
    return _resnet('resnet110', plan, num_classes, dense_classifier, pretrained)

def resnet1202(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(1202, 16)
    return _resnet('resnet1202', plan, num_classes, dense_classifier, pretrained)

# Wide ResNet Models
def wide_resnet20(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(20, 32)
    return _resnet('wide_resnet20', plan, num_classes, dense_classifier, pretrained)

def wide_resnet32(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(32, 32)
    return _resnet('wide_resnet32', plan, num_classes, dense_classifier, pretrained)

def wide_resnet44(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(44, 32)
    return _resnet('wide_resnet44', plan, num_classes, dense_classifier, pretrained)

def wide_resnet56(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(56, 32)
    return _resnet('wide_resnet56', plan, num_classes, dense_classifier, pretrained)

def wide_resnet110(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(110, 32)
    return _resnet('wide_resnet110', plan, num_classes, dense_classifier, pretrained)

def wide_resnet1202(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(1202, 32)
    return _resnet('wide_resnet1202', plan, num_classes, dense_classifier, pretrained)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock2(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(BasicBlock2, self).__init__()
        if norm_layer is None:
            norm_layer = layers_class.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = layers_class.Conv2d(inplanes, planes, 3, stride, padding=1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = layers_class.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck2(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        last_activation="relu",
    ):
        super(Bottleneck2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = layers_class.Conv2d(inplanes, width, 3, padding=1)
        self.bn1 = norm_layer(width)
        self.conv2 = layers_class.Conv2d(width, width, 3, stride, groups, padding=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = layers_class.Conv2d(width, planes * self.expansion, 3, padding=1)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if last_activation == "relu":
            self.last_activation = nn.ReLU(inplace=True)
        elif last_activation == "none":
            self.last_activation = nn.Identity()
        elif last_activation == "sigmoid":
            self.last_activation = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.last_activation(out)

        return out


class ResNet2(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_channels=3,
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        last_activation="relu",
        num_classes=10,
        img_dim=32,
    ):
        super(ResNet2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        # self._last_activation = last_activation

        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        print(f"resnet img_dim: {img_dim}")
        if img_dim <= 48:
            # CIFAR Stem
            self.stem = nn.Sequential()
            self.stem.add_module('conv0', layers_class.Conv2d(num_channels, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False))
            self.stem.add_module('BN1', norm_layer(num_out_filters))
            self.stem.add_module('ReLU1', nn.ReLU(inplace=True))
        else:
            self.stem = nn.Sequential()
            self.stem.add_module('conv0', layers_class.Conv2d(num_channels, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False))
            self.stem.add_module('BN1', norm_layer(num_out_filters))
            self.stem.add_module('ReLU1', nn.ReLU(inplace=True))
            self.stem.add_module('MaxPool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block,
            num_out_filters,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block,
            num_out_filters,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block,
            num_out_filters,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            last_activation=last_activation,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = layers_class.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layers_class.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (layers_class.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck2):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock2):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False, last_activation="relu"
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layers_class.Conv2d(self.inplanes, planes * block.expansion, 1, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                last_activation=(last_activation if blocks == 1 else "relu"),
            )
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    last_activation=(last_activation if i == blocks - 1 else "relu"),
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.padding(x)

        # old
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        fc_out = self.fc(x)

        return fc_out

def resnet34(input_shape, num_classes, dense_classifier=False, pretrained=False):
    return ResNet2(BasicBlock2, [3, 4, 6, 3], num_classes=num_classes)