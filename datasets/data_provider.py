from .data_list import ImageList
import torch.utils.data as util_data
from torchvision import transforms
from datasets import sampler
from datasets.sampler import N_Way_K_Shot_BatchSampler, TaskSampler, PseudoWeightedRandomSampler
from collections import Counter
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def get_dataloader_from_image_filepath(
        images_file_path,  batch_size=32, resize_size=256, is_train=True, crop_size=224, center_crop=True, args=None,
                     sampler=None, rand_aug_size=0, is_source=True):

    if images_file_path is None:
        return None, None

    data_sampler = None
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train is not True:  # eval mode
        transformer = transforms.Compose([
            ResizeImage(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize])

        images = ImageList(open(images_file_path).readlines(), transform=transformer, args=args)
        images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    else:  # training mode
        if center_crop:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(crop_size),
                                              transforms.ToTensor(),
                                              normalize])
        else:
            transformer = transforms.Compose([ResizeImage(resize_size),
                                              transforms.RandomCrop(crop_size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              normalize])

        images = ImageList(open(images_file_path).readlines(), transform=transformer, args=args, rand_aug_size=rand_aug_size)

        if sampler is None:
            images_loader = util_data.DataLoader(images, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
        elif sampler is not None:
            images_loader, data_sampler = select_training_dataloader(images, args, sampler, is_source)
        else:
            raise ValueError('could not create dataloader under the given config')

    return images_loader, data_sampler

def select_training_dataloader(images, args, sampler, is_source):
    if sampler == 'N_Way_K_Shot_BatchSampler':
        images_loader, data_sampler = nway_kshot_dataloader(images, args)
    elif sampler == 'ClassBalancedBatchSampler':
        images_loader, data_sampler = class_balanced_dataloader(images, args, is_source)
    else:
        images_loader, data_sampler = self_training_dataloader(images, args, sampler)
    return images_loader, data_sampler

def class_balanced_dataloader(images, args, is_source):
    if is_source:
        # use ground truth-label
        count_dict = Counter(images.labels)
        count_dict_full = {lbl: 0 for lbl in range(args.class_num)}
        for k, v in count_dict.items(): count_dict_full[k] = v

        count_dict_sorted = {k: v for k, v in sorted(count_dict_full.items(), key=lambda item: item[0])}
        class_sample_count = np.array(list(count_dict_sorted.values()))
        class_sample_count = class_sample_count / class_sample_count.max()
        class_sample_count += 1e-8

        weights = 1 / torch.Tensor(class_sample_count)
        sample_weights = [weights[l] for l in images.labels]
        sample_weights = torch.DoubleTensor(np.array(sample_weights))
        class_balanced_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        images_loader = util_data.DataLoader(images, sampler=class_balanced_sampler, \
                                                   batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    else:
        # use pseudo label
        sample_weights = torch.DoubleTensor(np.ones(len(images.labels)))
        class_balanced_sampler = PseudoWeightedRandomSampler(sample_weights, len(images.labels))
        images_loader = util_data.DataLoader(images, sampler=class_balanced_sampler, \
                                             batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    return images_loader, class_balanced_sampler


def nway_kshot_dataloader(images, args):
    task_sampler = TaskSampler(set(images.labels), args)
    n_way_k_shot_sampler = N_Way_K_Shot_BatchSampler(images.labels, args.train_steps, task_sampler)
    meta_loader = util_data.DataLoader(images, shuffle=False, batch_sampler=n_way_k_shot_sampler, num_workers=args.num_workers)
    return meta_loader, n_way_k_shot_sampler

def self_training_dataloader(images, args, _sampler):
    task_sampler = TaskSampler(set(images.labels), args)
    self_train_sampler_cls = getattr(sampler, _sampler)
    self_train_sampler = self_train_sampler_cls(args.train_steps, task_sampler, args)
    self_train_dataloader = util_data.DataLoader(images, shuffle=False, batch_sampler=self_train_sampler, num_workers=args.num_workers)
    return self_train_dataloader, self_train_sampler


class DataLoaderManager:
    def __init__(self, args):
        self.args = args
        self.train_source_loader, self.train_source_sampler = get_dataloader_from_image_filepath(
            args.source_path, args=args, batch_size=args.batch_size,
            sampler=args.train_source_sampler, center_crop=args.center_crop, rand_aug_size=args.source_rand_aug_size, is_source=True)
        self.train_target_loader, self.train_target_sampler = get_dataloader_from_image_filepath(
                args.target_path, args=args, batch_size=args.batch_size,
            sampler=args.train_target_sampler, center_crop=args.center_crop, rand_aug_size=args.target_rand_aug_size, is_source=False)

        self.val_source_loader, self.val_source_sampler = get_dataloader_from_image_filepath(
            args.source_path, args=args, batch_size=args.batch_size, is_train=False, is_source=True)
        self.val_target_loader, self.val_target_sampler = get_dataloader_from_image_filepath(
            args.target_path, args=args, batch_size=args.batch_size, is_train=False, is_source=False)

        if type(args.test_path) is list:
            tst_ls = []
            for tst_addr in args.test_path:
                tst_ls.append(list(get_dataloader_from_image_filepath(
                tst_addr, args=args, batch_size=args.batch_size, is_train=False, is_source=False)))
            self.test_loader, self.test_sampler = zip(*tst_ls)
        else:
            self.test_loader, self.test_sampler = get_dataloader_from_image_filepath(
            args.test_path, args=args, batch_size=args.batch_size, is_train=False, is_source=False)

    def update_sampler(self, model_instance, iter):
        if self.train_target_sampler is not None and hasattr(self.train_target_sampler, 'update'):
            self.train_target_sampler.update(model_instance, self.val_target_loader, iter)

        if self.train_source_sampler is not None and hasattr(self.train_source_sampler, 'update'):
            self.train_source_sampler.update(model_instance, self.val_source_loader, iter)


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter([])

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)