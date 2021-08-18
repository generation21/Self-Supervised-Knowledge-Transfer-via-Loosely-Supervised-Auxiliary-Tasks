import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='/raid/TTL/cifar10', type=str, help='Root of directory path of data'
    )
    parser.add_argument(
        '--dataset',
        default='cifar10', type=str, help='used dataset of cifar10 | cifar100 | imagenet | object detection | Instance segmentation'
    )
    parser.add_argument(
        '--n_class',
        default=10, type=int, help='Number of class'
    )
    parser.add_argument(
        '--n_source_class',
        default=10, type=int, help='Number of source class'
    )
    parser.add_argument(
        '--batch_size',
        default=128, type=int,
    )
    parser.add_argument(
        '--epochs',
        default=200, type=int,
    )
    parser.add_argument(
        '--model',
        default='resnet', type=str, help='student model (resnet | vgg | inception | dense)'
    )
    parser.add_argument(
        '--model_depth',
        default=18, type=int, help='model depth(resnet 18,50,101,152| vgg 16, 19 (bn)| inception | dense)'
    )
    parser.add_argument(
        '--source_arch',
        default='', type=str, help='places365 pretrained base model'
    )
    parser.add_argument(
        '--sourceKind',
        default='imagenet', type=str, help='places | cifar pretrained model'
    )
    parser.add_argument(
        '--lr',
        default=1e-3, type=float, help='learning rate'
    )
 
    parser.add_argument(
        '--result',
        default='/raid/video_data/output/result', type=str, help='output path'
    )
    parser.add_argument(
        '--save_model_path',
        default='model_ckp', type=str, help='save_model_path path'
    )

    parser.add_argument(
        '--pretrained_path',
        default='', type=str, help='pretrained model'
    )
    parser.add_argument(
        '--T',
        default=3, type=float, help='pretrained model'
    )
    parser.add_argument(
        '--alpha',
        default=0.1, type=float, help='pretrained model'
    )
    parser.add_argument(
        '--transfer_module',
        action='store_true')
    parser.set_defaults(transfer_module=False)


    parser.set_defaults(pretrained=False)
   
    parser.add_argument(
        '--isSource',
        action='store_true', help='Source Network is used'
    )
    parser.set_defaults(isSource=False)
    parser.add_argument(
        '--classifier_loss_method',
        default='cel',
        help='cel(cross entropy loss) | fl (focal loss)'
    )
    parser.add_argument(
        '--auxiliary_loss_method',
        default='cel',
        help='cel(cross entropy loss) | fl (focal loss)'
    )
    parser.add_argument(
        '--layer_num',
        default='b4',
        type=str,
        help='cel(cross entropy loss) | fl (focal loss)'
    )

    parser.add_argument('--optim', default="sgd", type=str,
                    help='optimizer : sgd | adam')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
    parser.add_argument('--multi_source', action='store_true')
    parser.set_defaults(multi_source=False)
    
    args = parser.parse_args()
    return args
    