import torch
from torch import nn
import torchvision
from models import small_resnet, resnet

def generate_model(opt):


    if opt.model == 'resnet20':
        model = small_resnet.resnet20(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet32':
        model = small_resnet.resnet32(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet44':
        model = small_resnet.resnet44(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet56':
        model = small_resnet.resnet56(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet110':
        model = small_resnet.resnet110(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet18':
        model = resnet.resnet18(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet34':
        model = resnet.resnet34(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet50':
        model = resnet.resnet50(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet101':
        model = resnet.resnet101(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
    elif opt.model == 'resnet152':
        model = resnet.resnet152(
                num_classes = opt.n_class,
                isSource = opt.isSource,
                sourceKind = opt.sourceKind,
                transfer_module = opt.transfer_module,
                multi_source = opt.multi_source)
        
    return model
