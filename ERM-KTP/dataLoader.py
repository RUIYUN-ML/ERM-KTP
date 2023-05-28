import os
import sys

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from seed import set_work_init_fn


class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.Train = train
        self.root_dir = root
        self.transform = transform
        self.train_dir = os.path.join(self.root_dir, "train")
        self.val_dir = os.path.join(self.root_dir, "val")

        if (self.Train):
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()

        self._make_dataset(self.Train)

        words_file = os.path.join(self.root_dir, "words.txt")
        wnids_file = os.path.join(self.root_dir, "wnids.txt")

        self.set_nids = set()

        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip("\n"))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split("\t")
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip("\n").split(","))[0]

    def _create_class_idx_dict_train(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(self.train_dir):
            for f in files:
                if f.endswith(".JPEG"):
                    num_images = num_images + 1

        self.len_dataset = num_images

        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(self.val_dir, "images")
        if sys.version_info >= (3, 5):
            images = [d.name for d in os.scandir(val_image_dir) if d.is_file()]
        else:
            images = [d for d in os.listdir(val_image_dir) if os.path.isfile(os.path.join(train_dir, d))]
        val_annotations_file = os.path.join(self.val_dir, "val_annotations.txt")
        self.val_img_to_class = {}
        set_of_classes = set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split("\t")
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])

        self.len_dataset = len(list(self.val_img_to_class.keys()))
        classes = sorted(list(set_of_classes))
        # self.idx_to_class = {i:self.val_img_to_class[images[i]] for i in range(len(images))}
        self.class_to_tgt_idx = {classes[i]: i for i in range(len(classes))}
        self.tgt_idx_to_class = {i: classes[i] for i in range(len(classes))}

    def _make_dataset(self, Train=True):
        self.images = []
        if Train:
            img_root_dir = self.train_dir
            list_of_dirs = [target for target in self.class_to_tgt_idx.keys()]
        else:
            img_root_dir = self.val_dir
            list_of_dirs = ["images"]

        for tgt in list_of_dirs:
            dirs = os.path.join(img_root_dir, tgt)
            if not os.path.isdir(dirs):
                continue

            for root, _, files in sorted(os.walk(dirs)):
                for fname in sorted(files):
                    if (fname.endswith(".JPEG")):
                        path = os.path.join(root, fname)
                        if Train:
                            item = (path, self.class_to_tgt_idx[tgt])
                        else:
                            item = (path, self.class_to_tgt_idx[self.val_img_to_class[fname]])
                        self.images.append(item)

    def return_label(self, idx):
        return [self.class_to_label[self.tgt_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img_path, tgt = self.images[idx]
        with open(img_path, 'rb') as f:
            sample = Image.open(img_path)
            sample = sample.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, tgt


class DataLoader():
    def __init__(self,dataset, batch_size, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed

    def load_data(self, img_size=32):
        data_dir = '/home/data/'
        data_transforms = {
            'cifar-train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
            'cifar-val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
            'imagenet-train':transforms.Compose([
                transforms.Resize(64),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'imagenet-val':transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
        }

        if self.dataset == 'cifar-10':
            data_train = datasets.CIFAR10(root=data_dir,
                                          transform=data_transforms['cifar-train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR10(root=data_dir,
                                         transform=data_transforms['cifar-val'],
                                         train=False,
                                         download=True)
        elif self.dataset == 'cifar-100':
            data_train = datasets.CIFAR100(root=data_dir,
                                          transform=data_transforms['cifar-train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR100(root=data_dir,
                                         transform=data_transforms['cifar-val'],
                                         train=False,
                                         download=True)
        elif self.dataset == 'imagenet':
            data_train = TinyImageNet(data_dir + 'tiny-imagenet-200', train=True, transform=data_transforms['imagenet-train'])
            data_test = TinyImageNet(data_dir + 'tiny-imagenet-200', train=False, transform=data_transforms['imagenet-val'])
            
        image_datasets = {'train': data_train, 'val': data_test}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                                           batch_size=self.batch_size, pin_memory=True,
                                                           shuffle=True, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)
        dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                         batch_size=self.batch_size, pin_memory=True,
                                                         shuffle=False, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        return dataloaders,dataset_sizes




class UnlearningDataLoader():
    def __init__(self,dataset, batch_size, num_unlearn, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_unlearn = num_unlearn
        self.seed = seed

    def load_data(self):
        data_dir = '/home/data/'
        torch.manual_seed(0)
        data_transforms = {
            'cifar-train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
            'cifar-val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
            'imagenet-train':transforms.Compose([
                transforms.Resize(64),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
            'imagenet-val':transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ]),
        }

        if self.dataset == 'cifar-10':
            data_train = datasets.CIFAR10(root=data_dir,
                                          transform=data_transforms['cifar-train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR10(root=data_dir,
                                         transform=data_transforms['cifar-val'],
                                         train=False,
                                         download=True)
        elif self.dataset == 'cifar-100':
            data_train = datasets.CIFAR100(root=data_dir,
                                          transform=data_transforms['cifar-train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR100(root=data_dir,
                                         transform=data_transforms['cifar-val'],
                                         train=False,
                                         download=True)
        elif self.dataset == 'imagenet':
            data_train = TinyImageNet(data_dir + 'tiny-imagenet-200', train=True, transform=data_transforms['imagenet-train'])
            data_test = TinyImageNet(data_dir + 'tiny-imagenet-200', train=False, transform=data_transforms['imagenet-val'])

        target_index = []
        nontarget_index = []
        for i in range(0, len(data_train)):
            if data_train[i][1] >= 0 and data_train[i][1] < self.num_unlearn:
                target_index.append(i)
            else:
                nontarget_index.append(i)

        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(data_train,
                                                         batch_size=self.batch_size, pin_memory=True,
                                                         shuffle=True, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)        
        dataloaders['unlearning'] = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, sampler = torch.utils.data.SubsetRandomSampler(target_index), 
                                                                pin_memory=True, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)
        dataloaders['remaining'] = torch.utils.data.DataLoader(data_train, batch_size=self.batch_size, sampler = torch.utils.data.SubsetRandomSampler(nontarget_index), 
                                                                pin_memory=True, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)
        dataloaders['val'] = torch.utils.data.DataLoader(data_test,
                                                         batch_size=self.batch_size, pin_memory=True,
                                                         shuffle=False, worker_init_fn=set_work_init_fn(self.seed), num_workers=16)
     
        dataset_sizes = {'train':len(data_train), 'unlearning':len(target_index), 'remaining':len(nontarget_index), 'val':len(data_test)}

        return dataloaders, dataset_sizes