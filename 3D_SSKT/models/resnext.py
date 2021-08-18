import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = ['ResNeXt', 'resnet50', 'resnet101']

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400,
                 isSource = False,
                 transfer_module = False,
                 n_source_class = 1000,
                 layer_num = 'b4',
                 multi_source = False):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
        self.isSource = isSource
        self.transfer_module = transfer_module
        self.multi_source = multi_source
        self.n_source_class = n_source_class
        if n_source_class == 1000:
            self.sourceKind = 'imagenet'
        elif n_source_class == 365:
            self.sourceKind = 'places'
        self.layer_num = layer_num
        if self.isSource: 
            if self.transfer_module:
                features_dim = [8, 4, 2, 1]
                in_features = int((cardinality * 32 * block.expansion) / features_dim[int((self.layer_num)[-1]) - 1])
                if multi_source:
                    self.auxiliary_imagenet = nn.Sequential(
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten(),
                        nn.Linear(in_features, 1000),
                    )  
                    self.auxiliary_places = nn.Sequential(
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten(),
                        nn.Linear(in_features, 365),
                    )
                else:
                    self.auxiliary = nn.Sequential(
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten(),
                        nn.Linear(in_features, n_source_class),)
            else:
                if self.multi_source:
                    self.auxiliary_imagenet = nn.Linear(cardinality * 32 * block.expansion, 1000)
                    self.auxiliary_places = nn.Linear(cardinality * 32 * block.expansion, 365)
                else:
                    self.auxiliary = nn.Linear(cardinality * 32 * block.expansion, n_source_class)
                
    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        pool = self.avgpool(b4)

        flatten = pool.view(pool.size(0), -1)
        fc = self.fc(flatten)
        
        if self.isSource:
            if self.transfer_module:
                if self.multi_source:
                    auxiliary_output1 = self.auxiliary_imagenet(vars()[self.layer_num])
                    auxiliary_output2 = self.auxiliary_places(vars()[self.layer_num])
                    return fc, auxiliary_output1, auxiliary_output2
                else: 
                    auxiliary_output = self.auxiliary(vars()[self.layer_num])
                    if self.sourceKind == 'imagenet':
                        return fc, auxiliary_output, None
                    elif self.sourceKind == 'places':
                        return fc, None, auxiliary_output
            else:
                if self.multi_source:
                    auxiliary_output1 = self.auxiliary_imagenet(flatten)
                    auxiliary_output2 = self.auxiliary_places(flatten)
                    return fc, auxiliary_output1, auxiliary_output2
                else:
                    auxiliary_output = self.auxiliary(flatten)
                    if self.sourceKind == 'imagenet':
                        return fc, auxiliary_output, None
                    elif self.sourceKind == 'places':
                        return fc, None, auxiliary_output

        else:
            return fc, None, None


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model
