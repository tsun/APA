import random
import numpy as np
import os, os.path as osp

import torch
import logging
import argparse
from collections import OrderedDict

import pickle
logger_init = False

def init_logger(_log_file, use_file_logger=True, dir='log/'):
    if not os.path.exists(dir):
        os.mkdir(dir)
    log_file = osp.join(dir, _log_file + '.log')
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d_%H.%M.%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)
    if use_file_logger:
        fhlr = logging.FileHandler(log_file)
        fhlr.setFormatter(formatter)
        logger.addHandler(fhlr)

    global logger_init
    logger_init = True

# set random number generators' seeds
def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def strlist(v):
    # just list
    if isinstance(v, list):
        return v

    # just string
    if '[' not in v or ']' not in v:
        return v

    v_list = v.strip('[]').split(',')
    v_list = [vi for vi in v_list]
    return v_list

def str2list(v):
    if isinstance(v, list):
        return v

    v_list = v.strip('[]').split(',')
    v_list = [int(vi) for vi in v_list]
    return v_list

def strlist2strint(l):
    return [int(_) for _ in l.split(',')]

def save_checkpoint(model, filename):
    weight_dicts = model.to_dicts()
    with open(filename, "wb" ) as fc:
        pickle.dump(weight_dicts, fc)

def load_checkpoint(model, filename):
    with open(filename, "rb" ) as fc:
        dicts = pickle.load(fc)
    try:
        model.from_dicts(dicts)
    except:
        new_dicts = []
        for _dict in dicts:
            new_dict = {}
            if isinstance(_dict, OrderedDict):
                for name, param in _dict.items():
                    namel = name.split('.')
                    key = '.'.join(namel[1:])
                    new_dict.update({key: param})
                new_dicts.append(new_dict)
            else:
                new_dicts.append(_dict)
        model.from_dicts(new_dicts)

def parse_address(address):
    if address is None:
        return ''

    return address.split('/')[-1].split('.')[0]

def args_to_str_src(args):
    return '_'.join([args.base_model, args.dataset, parse_address(args.source_path)]) # str(args.bottleneck_dim), str(args.center_crop)

def args_to_str(args):
    return '_'.join([args.model, args.dataset, parse_address(args.source_path), parse_address(args.target_path)]) # str(args.bottleneck_dim), str(args.center_crop)
