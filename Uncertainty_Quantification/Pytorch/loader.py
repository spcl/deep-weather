import os
import torch
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from utils import (
    reduce_sample_y,
    random_crop,
    horizontal_flip,
    vertical_flip,
    transpose,
)

TRANSFORMATION_DICTIONARY = {
    "RandomCrop": random_crop,
    "RandomHorizontalFlip": horizontal_flip,
    "RandomVerticalFlip": vertical_flip,
    "Transpose": transpose,
}


class Weatherdataset(Dataset):
    def __init__(self, args, step="train", infer=False):
        self.args = args
        directory = os.path.join(args.data_dir, step)
        data_list = os.listdir(directory)
        pattern_x = re.compile("inputLST[0-9]")
        pattern_y = re.compile("inputLSTC")
        self.datalist_x = [i for i in data_list if not (pattern_x.match(i) is None)]
        self.datalist_y = [i for i in data_list if not (pattern_y.match(i) is None)]
        self.datalist_x.sort()
        self.datalist_y.sort()

        self.length = len(self.datalist_x)

        if not infer:
            assert self.length == len(
                self.datalist_y
            ), "number of input samples has to match the number of output samples"

        self.infer = infer
        self.step = step

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        data_x = np.load(self.datalist_x[idx])
        data_x = np.reshape(
            data_x,
            [
                data_x.shape[0] * data_x.shape[1] * data_x.shape[2],
                data_x.shape[3],
                data_x.shape[4],
                data_x.shape[5],
            ],
        )
        # TODO use standardize maps here
        if not self.infer and self.step == "train":
            data_y = np.load(self.datalist_y[idx])
            data_y = reduce_sample_y(data_y, self.args)
            # TODO possibly standardize output here
            for aug in self.args.augmentation:  # Apply transformations if chosen
                data_x, data_y = TRANSFORMATION_DICTIONARY[aug](
                    data_x, data_y, self.args
                )

            return torch.from_numpy(data_x), torch.from_numpy(data_y)
        else:
            if self.step == "val" or self.step == "test":
                data_y = np.load(self.datalist_y[idx])
                data_y = reduce_sample_y(data_y, self.args)
                # TODO possibly standardize output here
                return torch.from_numpy(data_x), torch.from_numpy(data_y)
            return torch.from_numpy(data_x)


class WDatamodule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.train_dims = None

    def load_datasets(self, args):
        return (
            self.train_dataloader(args),
            self.val_dataloader(args),
            self.test_dataloader(args),
        )

    def setup(self, args):
        self.train, self.val, self.test = self.load_datasets(args)

    def train_dataloader(self, args):
        return DataLoader(
            Weatherdataset(args, step="train"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    def val_dataloader(self, args):
        return DataLoader(
            Weatherdataset(args, step="val"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self, args):
        return DataLoader(
            Weatherdataset(args, step="test"),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
