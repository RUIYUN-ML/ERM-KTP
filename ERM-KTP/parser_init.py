import argparse

def parser_init():
    parser = argparse.ArgumentParser(description='ERM-KTP')
    parser.add_argument('--name', default='test_model',
                        help='filename to output best model')  # save output
    parser.add_argument('--model', default='resnet20', help="models, resnet20/resnet50/resnext50")
    parser.add_argument('--dataset', default='cifar-10', help="datasets, cifar-10/cifar-100/imagenet")
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--num_unlearn', default=4, type=int)
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    parser.add_argument('--exp_dir', default='')
    parser.add_argument('--ifmask', default=True, type=bool,
                        help="whether use learnable mask (i.e. gate matrix)")
    parser.add_argument('--optim', default='sgd', type=str,
                        help="optimizer: adam | sgd")
    parser.add_argument('--lr', default=0.1, type=float,
                        help="learning rate for normal path")
    parser.add_argument('--train', default='True', type=str,
                        help='train or test the model')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed for the entire program')
    parser.add_argument('--cudnn_behavoir', default='benchmark', type=str,
                        help='cudnn behavoir [benchmark|normal(default)|slow|none] from left to right, cudnn randomness decreases, speed decreases')
    parser.add_argument('--load_checkpoint', default='',
                        type=str, help='path to load a checkpoint')

    return parser.parse_args()