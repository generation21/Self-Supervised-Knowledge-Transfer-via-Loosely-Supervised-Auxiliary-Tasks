import torch
from torch import nn

from models import resnet, resnext


def generate_model(opt):
    assert opt.model in [
        'resnet', 'resnext'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                isSource = opt.isSource,
                transfer_module = opt.transfer_module,
                sourceKind = opt.sourceKind,
                layer_num = opt.layer_num,
                multi_source = opt.multi_source)
    print(opt.no_cuda)
    print(type(opt.no_cuda))
    if not opt.no_cuda:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            print('loading pretrained model arch', pretrain['arch'], opt.arch)
            assert opt.arch == pretrain['arch']

            pretrained_dict = pretrain['state_dict']
            model_dict = model.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            pretrained_dict = {str.replace(k,'module.',''): v for k,v in pretrained_dict.items()}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            if opt.inference == False:
               
                model.module.fc = nn.Linear(model.module.fc.in_features,
                                            opt.n_finetune_classes)
                model.module.fc = model.module.fc.cuda()

                parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
                
                print(model)
                return model, parameters
            elif opt.inference:
                model = model.cuda()
                model = nn.DataParallel(model, device_ids=None)
                return model, model.parameters()
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            print('loading pretrained model arch', pretrain['arch'])
            pretrain = torch.load(opt.pretrain_path)
            
            assert opt.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])


            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            return model, parameters

    return model, model.parameters()

from models.SourceNet import SourceCNN, SourceImageNet, SourcePlaces
def SourceNetworkGeneration(opt):
    sourceImagenet = None
    sourcePlaces = None
    if opt.isSource:
        if opt.multi_source:
            sourceImagenet = SourceImageNet(opt).cuda()
            sourcePlaces = SourcePlaces(opt).cuda()
        elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
            sourceImagenet = SourceImageNet(opt).cuda()
        elif (not opt.multi_source) and opt.sourceKind == 'places':
            sourcePlaces = SourcePlaces(opt).cuda()
        if torch.cuda.device_count() > 1:
            if opt.multi_source:
                sourceImagenet = nn.DataParallel(sourceImagenet)
                sourcePlaces = nn.DataParallel(sourcePlaces)
            elif (not opt.multi_source) and opt.sourceKind == 'imagenet':
                sourceImagenet = nn.DataParallel(sourceImagenet)
            elif (not opt.multi_source) and opt.sourceKind == 'places':
                sourcePlaces = nn.DataParallel(sourcePlaces)
    return sourceImagenet, sourcePlaces