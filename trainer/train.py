import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)

from utils.utils import *
from trainer.argument_parser import argument_parse
from trainer.evaluate import evaluate_all

from datasets.data_provider import DataLoaderManager, ForeverDataIterator
from datasets.sampler import TaskSampler
from importlib import import_module

from torch.optim.lr_scheduler import LambdaLR

from trainer.losses import get_class_criterion

# repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train_source(model_instance, train_source_loader, val_source_loader, optimizer, lr_scheduler=None, args=None, test_loader=None):
    model_instance.set_train(True)
    print("start train source model...")
    iter_num = 0
    epoch = 0
    max_iter = args.train_steps
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter, initial=0)

    iter_train_source_loader = ForeverDataIterator(train_source_loader)
    while True:
        for _ in tqdm.tqdm(
                range(max_iter),
                total=len(train_source_loader),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):

            datas = next(iter_train_source_loader)
            inputs_source, labels_source, indexes_source, _ = datas

            inputs_source = inputs_source.cuda()
            labels_source = labels_source.cuda()

            optimizer.zero_grad()
            outputs_source = model_instance.forward(inputs_source)

            # classification loss
            class_criterion = get_class_criterion(args)
            classifier_loss = class_criterion(outputs_source, labels_source)
            classifier_loss.backward()
            optimizer.step()

            if iter_num % args.eval_interval == 0 and iter_num != 0:
                evaluate_all(model_instance, val_source_loader, None, test_loader, iter_num, args)

            if (iter_num % args.save_interval == 0 and iter_num != 0) and args.save_checkpoint:
                checkpoint_name = 'checkpoint_base/' + args_to_str_src(args) + '_' + args.timestamp + '_' + str(args.random_seed) + '_' + str(iter_num) + '.pth'
                save_checkpoint(model_instance, checkpoint_name)
                logging.info('Train iter={}:Checkpoint saved to {}'.format(iter_num, checkpoint_name))

            iter_num += 1

            total_progress_bar.update(1)
            if iter_num > max_iter:
                break
        epoch += 1
        if iter_num > max_iter:
            break

    print('finish source train')



def train(model_instance, train_source_loader, val_source_loader, train_target_loader, val_target_loader,
          optimizer, lr_scheduler, args=None, data_loader_manager=None, test_loader=None):
    model_instance.set_train(True)
    print("start train...")
    iter_num = args.restore_iter
    epoch = 0
    max_iter = args.train_steps
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter, initial=args.restore_iter)

    iter_train_source_loader = ForeverDataIterator(train_source_loader)
    iter_train_target_loader = ForeverDataIterator(train_target_loader)

    while True:
        for _ in tqdm.tqdm(
                range(max_iter-args.restore_iter),
                total=min(len(train_source_loader), len(train_target_loader)),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):

            if iter_num % args.yhat_update_freq == 0:
                data_loader_manager.update_sampler(model_instance, iter_num)

            datas = next(iter_train_source_loader)
            datat = next(iter_train_target_loader)

            inputs_source, labels_source, indexes_source, inputs_source_rand_aug = datas
            inputs_target, labels_target, indexes_target, inputs_target_rand_aug = datat

            inputs_source = inputs_source.cuda()
            inputs_target = inputs_target.cuda()
            labels_source = labels_source.cuda()
            labels_target = labels_target.cuda()
            inputs_source_rand_aug = [_.cuda() for _ in inputs_source_rand_aug]
            inputs_target_rand_aug = [_.cuda() for _ in inputs_target_rand_aug]

            optimizer.zero_grad()

            total_loss = model_instance.get_loss(inputs_source, inputs_target, labels_source, labels_target,
                                                 inputs_source_rand_aug, inputs_target_rand_aug, args)
            total_loss.backward()
            optimizer.step()

            if iter_num % args.lr_scheduler_rate == 0:
                lr_scheduler.step()

            if iter_num % args.eval_interval == 0 and iter_num != 0:
                evaluate_all(model_instance, val_source_loader, val_target_loader, test_loader, iter_num, args)


            if (iter_num % args.save_interval == 0 and iter_num != 0) and args.save_checkpoint:
                checkpoint_name = 'checkpoint/'+args_to_str(args)+'_'+args.timestamp+'_'+ str(args.random_seed)+'_'+str(iter_num)+'.pth'
                save_checkpoint(model_instance, checkpoint_name)
                logging.info('Train iter={}:Checkpoint saved to {}'.format(iter_num, checkpoint_name))

                if args.save_optimizer:
                    optimizer_checkpoint_name = 'checkpoint/' + args_to_str(args) + '_' + args.timestamp + '_' + str(args.random_seed)+ '_' + str(
                        iter_num) + '_optimizer.pth'
                    torch.save(optimizer.state_dict(), optimizer_checkpoint_name)

            iter_num += 1

            total_progress_bar.update(1)
            if iter_num > max_iter:
                break
        epoch += 1
        if iter_num > max_iter:
            break
    print('finish train')


def _init_(args_, header):
    args = argument_parse(args_)

    # clear TaskSampler state for repeated experiments
    TaskSampler.__del__()

    resetRNGseed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dir = '{}_{}'.format(args.timestamp, '_'.join([_ for _ in [args.model or args.base_model, parse_address(args.source_path),
                                                   parse_address(args.target_path)] if _!='']))

    if args.train_source_sampler is not None:
        dir += '_src_{}'.format(args.train_source_sampler)
    if args.train_target_sampler is not None:
        dir += '_tgt_{}'.format(args.train_target_sampler)

    if not logger_init:
        init_logger(dir, args.use_file_logger, args.log_dir)


    logging.info(header)
    logging.info(args)

    return args

def train_source_main(args_, header=''):
    args = _init_(args_, header)

    assert 'Base' in args.base_model
    try:
        model_module = import_module('model.'+args.base_model)
        Model = getattr(model_module, args.base_model)
        model_instance = Model(base_net=args.base_net, use_bottleneck=args.use_bottleneck, bottleneck_dim=args.bottleneck_dim, use_gpu=True, class_num=args.class_num, args=args)
    except:
        raise NotImplementedError('Unsupported model')

    data_loader_manager = DataLoaderManager(args)
    train_source_loader = data_loader_manager.train_source_loader
    val_source_loader = data_loader_manager.val_source_loader
    test_loader = data_loader_manager.test_loader

    param_groups = model_instance.get_parameter_list()

    optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.lr_momentum, weight_decay=args.lr_wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_scheduler_gamma * float(x)) ** (-args.lr_scheduler_decay_rate))

    train_source(model_instance, train_source_loader, val_source_loader, optimizer=optimizer, lr_scheduler=lr_scheduler, args=args,
          test_loader=test_loader)


def train_main(args_, header=''):
    args = _init_(args_, header)

    # get base model
    try:
        model_module = import_module('model.' + args.base_model)
        Model = getattr(model_module, args.base_model)
        source_model_instance = Model(base_net=args.base_net, use_bottleneck=args.use_bottleneck, bottleneck_dim=args.bottleneck_dim,
                 use_gpu=True, class_num=args.class_num, args=args)
    except:
        raise NotImplementedError('Unsupported model')

    if args.source_checkpoint is not None:
        load_checkpoint(source_model_instance, args.source_checkpoint)
    else:
        logging.info("training from scratch")

    #  get adaptation model
    try:
        model_module = import_module('model.'+args.model)
        Model = getattr(model_module, args.model)
        model_instance = Model(base_net=args.base_net, use_bottleneck=args.use_bottleneck, bottleneck_dim=args.bottleneck_dim,
                               use_gpu=True, class_num=args.class_num, args=args)
    except:
        raise NotImplementedError('unsupported model')

    model_instance.copy_from_source(source_model_instance)

    data_loader_manager = DataLoaderManager(args)
    train_source_loader, train_target_loader = data_loader_manager.train_source_loader, data_loader_manager.train_target_loader
    val_source_loader, val_target_loader = data_loader_manager.val_source_loader, data_loader_manager.val_target_loader
    test_loader = data_loader_manager.test_loader

    param_groups = model_instance.get_parameter_list()

    optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.lr_momentum, weight_decay=args.lr_wd, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_scheduler_gamma * float(x)) ** (-args.lr_scheduler_decay_rate))


    if args.restore_checkpoint is not None:
        load_checkpoint(model_instance, args.restore_checkpoint)
        args.restore_iter = int(args.restore_checkpoint.split('_')[-1].split('.')[0])
        logging.info('model weights restored from: {} at iter {}'.format(args.restore_checkpoint, args.restore_iter))
        model_instance.iter_num = args.restore_iter

        optimizer_checkpoint_name = args.restore_checkpoint.split('.')[0]+'_optimizer.pth'
        if os.path.exists(optimizer_checkpoint_name):
            optimizer.load_state_dict(torch.load(optimizer_checkpoint_name))

    train(model_instance, train_source_loader, val_source_loader, train_target_loader, val_target_loader,
          optimizer=optimizer, lr_scheduler=lr_scheduler, args=args,
          data_loader_manager = data_loader_manager, test_loader=test_loader)