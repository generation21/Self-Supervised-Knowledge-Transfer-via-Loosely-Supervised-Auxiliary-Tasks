import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
def branchBottleNeck(channel_in, channel_out, kernel_size, stride):
    middle_channel = channel_out//2
    return nn.Sequential(
        nn.Conv3d(channel_in, middle_channel, kernel_size=1, stride=1),
        nn.BatchNorm3d(middle_channel),
        nn.ReLU(),
        
        nn.Conv3d(middle_channel, middle_channel, kernel_size=kernel_size, stride=stride, padding=1),
        nn.BatchNorm3d(middle_channel),
        nn.ReLU(),
        
        nn.Conv3d(middle_channel, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm3d(channel_out),
        nn.ReLU(),
        )
class aggregation_feature(nn.Module):
    def __init__(self, in_features, out):
        super(aggregation_feature, self).__init__()
        features_dim = [8, 4, 2, 1]
        b1_in_features = in_features // 8 
        # print("---------------------------------------")
        # print(b1_in_features)
        # print("___________________________________")
        b2_in_features = in_features // 4 
        b3_in_features = in_features // 2 
        b4_in_features = in_features // 1
        self.b1_feature = branchBottleNeck(b1_in_features, 1000, kernel_size=3, stride=2)
        self.b2_feature = branchBottleNeck(b2_in_features, 1000, kernel_size=3, stride=2)
        self.b3_feature = branchBottleNeck(b3_in_features, 1000, kernel_size=3, stride=2)
        self.b4_feature = branchBottleNeck(b4_in_features, 1000, kernel_size=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(1000, out)
    def forward(self, b1, b2, b3, b4):
        # print(b1.size())
        x1 = self.b1_feature(b1)
        x1 = self.avgpool(x1)
        x2 = self.b2_feature(b2)
        x2 = self.avgpool(x2)
        x3 = self.b3_feature(b3)
        x3 = self.avgpool(x3)
        x4 = self.b4_feature(b4)
        x4 = self.avgpool(x4)
        
        x = (x1 + x2 + x3 + x4) / 4
        x = x.view(x.size(0), -1)
        x = self.linear(x)
            
        return x

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400,
                 isSource = False,
                 transfer_module = False,
                 sourceKind = "imagenet",
                 layer_num = 'b4',
                 multi_source = False):
        self.inplanes = 64
        super(ResNet, self).__init__()
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
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.isSource = isSource
        self.transfer_module = transfer_module
        self.multi_source = multi_source
        
        self.sourceKind = sourceKind
        self.layer_num = layer_num
        if self.sourceKind == 'imagenet':
            self.n_source_class = 1000
        elif self.sourceKind == 'places365':
            self.n_source_class = 365
        if self.isSource: 
            if self.transfer_module:
                features_dim = [8, 4, 2, 1]
                in_features = int((512 * block.expansion) / features_dim[int((self.layer_num)[-1]) - 1])
                if self.multi_source:
                    self.auxiliary_imagenet = aggregation_feature(512 * block.expansion, 1000)
                    self.auxiliary_places = aggregation_feature(512 * block.expansion, 365)
                else:
                    self.auxiliary = aggregation_feature(512 * block.expansion, self.n_source_class)
                #     self.auxiliary_imagenet = nn.Sequential(
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten(),
                #         nn.Linear(in_features, 1000),
                #     )  
                #     self.auxiliary_places = nn.Sequential(
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten(),
                #         nn.Linear(in_features, 365),
                #     )
                # else:
                #     self.auxiliary = nn.Sequential(
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.Conv3d(in_features, in_features, kernel_size=3, stride=1, padding=1),
                #         nn.BatchNorm3d(in_features), nn.ReLU(inplace=True),
                #         nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten(),
                #         nn.Linear(in_features, self.n_source_class),)  
            else:
                if self.multi_source:
                    self.auxiliary_imagenet = nn.Linear(512 * block.expansion, 1000)
                    self.auxiliary_places = nn.Linear(512 * block.expansion, 365)
                else:
                    self.auxiliary = nn.Linear(512 * block.expansion, self.n_source_class)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
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
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
                    auxiliary_output1 = self.auxiliary_imagenet(b1, b2, b3, b4)#vars()[self.layer_num])
                    auxiliary_output2 = self.auxiliary_places(b1, b2, b3, b4)#vars()[self.layer_num])
                    return fc, auxiliary_output1, auxiliary_output2
                else: 
                    auxiliary_output = self.auxiliary(b1, b2, b3, b4)#vars()[self.layer_num])
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


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
