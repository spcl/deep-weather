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
    standardize,
)

TRANSFORMATION_DICTIONARY = {
    "RandomCrop": random_crop,
    "RandomHorizontalFlip": horizontal_flip,
    "RandomVerticalFlip": vertical_flip,
    "Transpose": transpose,
}


class WeatherDataset(Dataset):
    def __init__(self, args, step="train", infer=False, year_dict=None):
        self.args = args
        data_list = os.listdir(args.data_directory)
        pat_x = ["inputLST" + i for i in year_dict[step]]
        pat_y = ["inputLSTC" + i for i in year_dict[step]]
        self.datalist_x = [
            os.path.join(args.data_directory, i)
            for i in data_list
            if any(j in i for j in pat_x)
        ]
        self.datalist_y = [
            os.path.join(args.data_directory, i)
            for i in data_list
            if any(j in i for j in pat_y)
        ]
        self.means = np.load(os.path.join(args.std_folder, "means.npy")).astype(
            np.float32
        )
        self.stddevs = np.load(os.path.join(args.std_folder, "stddevs.npy")).astype(
            np.float32
        )
        self.means = self.means[
            0,
            None,
            args.parameters.index(args.pred_type),
            :,
            : self.args.max_lat,
            : self.args.max_lon,
        ]
        self.stddevs = self.stddevs[
            0,
            None,
            args.parameters.index(args.pred_type),
            :,
            : self.args.max_lat,
            : self.args.max_lon,
        ]
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
        # crop to work with 5 pooling operations
        data_x = data_x[:, :, : self.args.max_lat, : self.args.max_lon]
        data_x = standardize(data_x, self.means, self.stddevs)
        if not self.infer and self.step == "train":
            data_y = np.load(self.datalist_y[idx])
            data_y = reduce_sample_y(data_y, self.args)
            # crop to work with 5 pooling operations
            data_y = data_y[:, :, : self.args.max_lat, : self.args.max_lon]
            data_y = standardize(data_y, self.means, self.stddevs)
            for aug in self.args.augmentation:  # Apply transformations if chosen
                data_x, data_y = TRANSFORMATION_DICTIONARY[aug](
                    data_x, data_y, self.args
                )

            return torch.from_numpy(data_x.copy()), torch.from_numpy(data_y.copy())
        else:
            if self.step == "val" or self.step == "test":
                data_y = np.load(self.datalist_y[idx])
                data_y = reduce_sample_y(data_y, self.args)
                # crop to work with 5 pooling operations
                data_y = data_y[:, :, : self.args.max_lat, : self.args.max_lon]
                data_y = standardize(data_y, self.means, self.stddevs)
                return torch.from_numpy(data_x), torch.from_numpy(data_y)
            return torch.from_numpy(data_x)


class WDatamodule(pl.LightningDataModule):
    def __init__(self, args, year_dict=None):
        super().__init__()
        self.train_dims = None
        self.year_dict = year_dict

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
            WeatherDataset(args, step="train", year_dict=self.year_dict),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

    def val_dataloader(self, args):
        return DataLoader(
            WeatherDataset(args, step="val", year_dict=self.year_dict),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

    def test_dataloader(self, args):
        return DataLoader(
            WeatherDataset(args, step="test", year_dict=self.year_dict),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
