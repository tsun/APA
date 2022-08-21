import argparse
from utils.utils import str2bool, strlist
import time

def argument_parse(args_):
    _timestamp = time.strftime("%Y-%m-%d_%H.%M.%S", time.localtime())
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model', default=None, type=str,
                        help='which model')
    parser.add_argument('--base_model', default='Base', type=str,
                        help='which model')
    parser.add_argument('--base_net', default='ResNet50', type=str,
                        help='base_net')
    parser.add_argument('--source_checkpoint', default=None, type=str,
                        help='checkpoint to restore weights')
    parser.add_argument('--restore_checkpoint', default=None, type=str,
                        help='checkpoint to restore weights')
    parser.add_argument('--restore_iter', default=0, type=int, help='restored training iteration')

    # dataset
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--datasets_dir', default='./dataset', type=str,
                        help='dataset dir')
    parser.add_argument('--source_path', default=None, type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--target_path', default=None, type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--test_path', default=None, type=strlist,
                        help='address of image list of test dataset')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='number of worker')
    parser.add_argument('--rand_aug_n', default=3, type=int,
                        help='number of rand aug transform')
    parser.add_argument('--rand_aug_m', default=2.0, type=float,
                        help='number of rand aug severity')
    parser.add_argument('--source_rand_aug_size', default=0, type=int,
                        help='number of rand aug data')
    parser.add_argument('--target_rand_aug_size', default=0, type=int,
                        help='number of rand aug data')

    # training configuration
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='config/dann.yml')
    parser.add_argument('--save_interval', default='5000', type=int,
                        help='save checkpoint interval')
    parser.add_argument('--eval_interval', default='200', type=int,
                        help='evaluate test accuracy interval')
    parser.add_argument('--train_source_steps', default=10000, type=int, help='number of training steps')
    parser.add_argument('--save_source_interval', default=10000, type=int, help='save checkpoint interval')
    parser.add_argument('--train_steps', default=20000, type=int, help='number of training steps')
    parser.add_argument('--iter', default=-1, type=int, help='current training iteration')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--class_num', default=-1, type=int, help='class number')

    parser.add_argument('--eval_source', default='True', type=str2bool,
                        help='whether evaluate source data')
    parser.add_argument('--eval_target', default='True', type=str2bool,
                        help='whether evaluate target data')
    parser.add_argument('--eval_test', default='True', type=str2bool,
                        help='whether evaluate extra data')

    parser.add_argument('--save_checkpoint', default='True', type=str2bool,
                        help='whether save checkpoint')
    parser.add_argument('--save_optimizer', default='False', type=str2bool,
                        help='whether save optimizer state')

    parser.add_argument('--lr', default=0.001, type=float,
                        help='lr')
    parser.add_argument('--lr_momentum', default=0.9, type=float,
                        help='lr schedule momentum')
    parser.add_argument('--lr_wd', default=0.0005, type=float,
                        help='lr weight decay')
    parser.add_argument('--lr_scheduler_gamma', default=0.0001, type=float,
                        help='lr scheduler gamma')
    parser.add_argument('--lr_scheduler_decay_rate', default=0.75, type=float,
                        help='lr schedule decay rate')
    parser.add_argument('--lr_scheduler_rate', default=1, type=int,
                        help='lr schedule rate')

    # environment
    parser.add_argument('--gpu_id', default='0', type=str,
                        help='which gpu to use')
    parser.add_argument('--random_seed', default='0', type=int,
                        help='random seed')
    parser.add_argument('--timestamp', default=_timestamp, type=str,
                        help='timestamp')

    # logger
    parser.add_argument('--use_file_logger', default='True', type=str2bool,
                        help='whether use file logger')
    parser.add_argument('--log_dir', default='log', type=str,
                        help='log directory')

    # data sampling
    parser.add_argument('--train_source_sampler', type=str, default=None,
                        help='sampler for self training, e.g., SelfTrainingVannilaSampler')

    parser.add_argument('--train_target_sampler', type=str, default=None,
                        help='sampler for self training, e.g., SelfTrainingVannilaSampler')

    parser.add_argument('--n_way', default=31, type=int,
                        help='number of classes for each classification task')
    parser.add_argument('--k_shot', default=1, type=int,
                        help='number of examples per class per domain, default using all examples available.')

    parser.add_argument('--yhat_update_freq', type=int, default=100,
                        help='frequency to update the self-training predictions on the target domain')
    parser.add_argument('--confidence_threshold', type=float, default=None, help='threshold for conditional sampling')
    parser.add_argument('--balance_domain', default='True', type=str2bool,
                        help='whether balance src and tgt data number')

    parser.add_argument('--center_crop', default=None, type=str2bool,
                        help='whether to center crop data')

    # model
    parser.add_argument('--bottleneck_dim', default=None, type=int, help="the dim of the bottleneck layer")
    parser.add_argument('--use_bottleneck', default='True', type=str2bool,
                        help='whether use bottleneck layer')
    parser.add_argument('--use_bottleneck_dropout', default='False', type=str2bool,
                        help='whether use bottleneck dropout layer')
    parser.add_argument('--use_dropout', default='False', type=str2bool,
                        help='whether use dropout layer')
    parser.add_argument('--use_hidden_layer', default='False', type=str2bool,
                        help='whether use hidden layer')

    parser.add_argument('--l2_normalize', default='True', type=str2bool,
                        help='whether use l2_normalize')
    parser.add_argument('--temperature', default=0.05, type=float,
                        help='temperature')

    # classification loss
    parser.add_argument('--class_criterion', default='CrossEntropyLoss', type=str,
                        help='loss for classification')
    parser.add_argument('--self_training_loss_weight', default='0.0', type=float,
                        help='self training loss weight')
    parser.add_argument('--self_training_conf', default='0.75', type=float,
                        help='self training confidence')

    # VAT loss
    parser.add_argument('--VAT_xi', default='10', type=float,
                        help='VAT_xi')
    parser.add_argument('--VAT_eps', default='1.0', type=float,
                        help='VAT_eps')
    parser.add_argument('--VAT_iter', default='1', type=int,
                        help='VAT_iter')
    parser.add_argument('--vat_loss_weight', default='0.0', type=float,
                        help='weight of vat loss')

    # fixmatch loss
    parser.add_argument('--fixmatch_loss_threshold', default='0.97', type=float,
                        help='fixmatch_loss_threshold')
    parser.add_argument('--fixmatch_loss_weight', default='0.0', type=float,
                        help='fixmatch_loss_weight')

    parser.add_argument('--save_eval_result', default='False', type=str2bool,
                        help='whether to save eval result to file')

    args = parser.parse_args(args_)

    # -----------------------------------------
    if args.dataset == 'Office-31':
        class_num = 31
        bottleneck_dim = 1024
        center_crop = False
    elif args.dataset == 'Office-Home':
        class_num = 65
        bottleneck_dim = 2048
        center_crop = False

        # Another choice for Office-home:
        # bottleneck_dim = 1024
        # MDD_margin = 3
        # center_crop = True
    elif args.dataset == 'visda':  # this follows Transfer-Learning-Libarary
        class_num = 12
        bottleneck_dim = 1024
        center_crop = True
    elif args.dataset == 'digits':
        class_num = 10
        bottleneck_dim = 128
        center_crop = False
        args.base_net = "DigitFeatures"
    elif args.dataset == 'DomainNet':
        class_num = 40
        bottleneck_dim = 2048
        center_crop = False
    else:
        raise NotImplementedError('Unsupported dataset')

    args.bottleneck_dim = bottleneck_dim if args.bottleneck_dim is None and args.use_bottleneck else args.bottleneck_dim
    args.center_crop = center_crop if args.center_crop is None else args.center_crop
    args.class_num = class_num

    return args