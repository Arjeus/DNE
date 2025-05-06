# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import json
import os
import warnings
import numpy as np
import torch
import lmdb
import pickle
import six
from PIL import Image
import pdb

from sklearn.datasets import load_digits
from torch.utils.data import Dataset
import torchvision.transforms as T

from continuum import ClassIncremental
from continuum.datasets import CIFAR100, ImageNet100, ImageFolderDataset, InMemoryDataset
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torchvision.transforms import functional as Fv

from sklearn.datasets import load_iris
from PIL import Image

try:
    interpolation = Fv.InterpolationMode.BICUBIC
except:
    interpolation = 3


class ImageFolderLMDB(torch.utils.data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = pickle.loads(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        im2arr = np.array(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return im2arr, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
    
class IrisAsCifar(InMemoryDataset):
    """Tabular Iris → 32×32 RGB images that mimic CIFAR‑100."""
    def __init__(
        self,
        train: bool = True,
        test_split: float = 0.5,
        seed: int = 0,
        transform=None,
    ):
        # 1 Load the 150×4 tabular iris data
        iris = load_iris()                                      # 3 classes, 150 samples :contentReference[oaicite:1]{index=1}
        X, y = iris.data.astype(np.float32), iris.target
        n_samples, n_features = X.shape                         # 150, 4

        # 2 Compute global min/max per feature for stable scaling
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        span = np.where(maxs > mins, maxs - mins, 1.0)          # avoid /0

        # 3 Stratified 50 % / 50 % split to stay reproducible
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_samples)
        split = int(n_samples * test_split)
        idxs = perm[split:] if train else perm[:split]
        X, y = X[idxs], y[idxs]

        # 4 Encode every sample into a 32×32×3 uint8 image
        imgs = np.zeros((len(X), 32, 32, 3), dtype=np.uint8)
        for i, row in enumerate(X):
            for f in range(n_features):
                # scale 0‑255
                v = int((row[f] - mins[f]) / span[f] * 255)
                r0, r1 = 8 * f, 8 * (f + 1)                     # stripe rows
                imgs[i, r0:r1, :, :] = v                       # broadcast

        # 5 Default TorchVision transforms (can be overridden)
        self.transform = transform or T.Compose([
            T.ToPILImage(),             # uint8 → PIL
            T.ToTensor(),               # → float32 [0,1]
            T.Normalize((0.5,)*3, (0.5,)*3),
        ])

        # 6 Call Continuum’s constructor
        super().__init__(x=imgs, y=y)

    # Override only to inject TorchVision transforms
    def __getitem__(self, index):
        img, label, _ = super().__getitem__(index)  # img is uint8 HWC
        img = self.transform(img)
        return img, label
    
class DigitsAsCifar(InMemoryDataset):
    """scikit‑learn Digits, shaped like CIFAR‑100 and usable by Continuum."""
    def __init__(
        self,
        train: bool = True,
        test_split: float = 0.5,
        seed: int = 0,
        transform=None,
    ):
        # ------------------------------------------------------------------
        # 1. load the raw 8×8 greyscale images
        # ------------------------------------------------------------------
        ds = load_digits()                                    # (1797, 8, 8)
        imgs8, labels = ds.images, ds.target

        # ------------------------------------------------------------------
        # 2. stratified 50 %/50 % split – keeps CL experiments reproducible
        # ------------------------------------------------------------------
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(imgs8))
        split = int(len(imgs8) * test_split)
        idxs = order[split:] if train else order[:split]

        # ------------------------------------------------------------------
        # 3. upscale 8×8→32×32 and copy the single channel → RGB
        #    Continuum expects H×W×C uint8 in [0, 255]
        # ------------------------------------------------------------------
        x = np.kron(imgs8[idxs], np.ones((4, 4)))             # 32×32 grey
        x = (x * 255 / 16).astype(np.uint8)[..., None]        # scale + channel
        x = np.repeat(x, 3, axis=3)                          # → RGB
        y = labels[idxs]

        # ------------------------------------------------------------------
        # 4. optional TorchVision transform pipeline
        # ------------------------------------------------------------------
        self.transform = transform or T.Compose([
            T.ToPILImage(),
            T.ToTensor(),                                     # 0‑1 float32
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # ------------------------------------------------------------------
        # 5. call the real Continuum constructor
        # ------------------------------------------------------------------
        super().__init__(x=x, y=y)    # InMemoryDataset stores the arrays

    # ----------------------------------------------------------------------
    #  The parent class already implements `get_data()`, but we overload
    #  `__getitem__` only to plug in our TorchVision transforms.
    # ----------------------------------------------------------------------
    def __getitem__(self, index):
        img, label, _ = super().__getitem__(index)            # img is uint8
        img = self.transform(img)                             # Tensor
        return img, label


class ImageNet1000(ImageFolderDataset):
    """Continuum dataset for datasets with tree-like structure.
    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
            self,
            data_path: str,
            train: bool = True,
            download: bool = False,
    ):
        super().__init__(data_path=data_path, train=train, download=download)

    def get_data(self):
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")
        return super().get_data()


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set.lower() == 'cifar':
        dataset = CIFAR100(args.data_path, train=is_train, download=True)
    elif args.data_set.lower() == 'imagenet100':
        dataset = ImageNet100(
            args.data_path, train=is_train,
            data_subset=os.path.join('./imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
        )
    elif args.data_set.lower() == 'digits':
        # replace CIFAR with Digits transparently
        dataset = DigitsAsCifar(train=is_train)
    
    elif args.data_set.lower() == 'iris':
        # replace CIFAR with Digits transparently
        dataset = IrisAsCifar(train=is_train)

    elif args.data_set.lower() == 'imagenet1000':
        dataset = ImageNet1000(args.data_path, train=is_train)
    else:
        raise ValueError(f'Unknown dataset {args.data_set}.')

    scenario = ClassIncremental(
        dataset,
        initial_increment=args.initial_increment,
        increment=args.increment,
        transformations=transform.transforms,
        class_order=args.class_order
    )
    nb_classes = scenario.nb_classes

    return scenario, nb_classes


def build_transform(is_train, args):
    if args.aa == 'none':
        args.aa = None

    with warnings.catch_warnings():
        resize_im = args.input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=args.input_size,
                is_training=True,
                color_jitter=args.color_jitter,
                auto_augment=args.aa,
                interpolation='bicubic',
                re_prob=args.reprob,
                re_mode=args.remode,
                re_count=args.recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    args.input_size, padding=4)

            if args.input_size == 32 and args.data_set == 'CIFAR':
                transform.transforms[-1] = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            return transform

        t = []
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                transforms.Resize(size, interpolation=interpolation),  # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.input_size))

        t.append(transforms.ToTensor())
        if args.input_size == 32 and args.data_set == 'CIFAR':
            # Normalization values for CIFAR100
            t.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        else:
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

        return transforms.Compose(t)
