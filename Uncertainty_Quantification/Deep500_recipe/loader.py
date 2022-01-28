import os

import torch
import numpy as np
import random
import re
from torch.utils.data import Dataset
from deep500.lv2.dataset import Dataset as D500Dataset
from deep500.lv2.sampler import ShuffleSampler
from utils import UQDataClass

from utils import (
    reduce_sample_y,
    reduce_sample_x,
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

    def __init__(self,
                 step="train",
                 infer=False,
                 year_dict=None,
                 args: UQDataClass = None):
        self.args = args
        data_list = os.listdir(args.data_directory)
        pat_x = ["inputLST" + i for i in year_dict[step]]
        pat_y = ["inputLSTC" + i for i in year_dict[step]]
        self.datalist_x = [
            os.path.join(args.data_directory, i) for i in data_list
            if any(j in i for j in pat_x)
        ]
        self.datalist_y = [
            os.path.join(args.data_directory, i) for i in data_list
            if any(j in i for j in pat_y)
        ]
        self.means = np.load(os.path.join(args.std_folder,
                                          "means.npy")).astype(np.float32)
        self.stddevs = np.load(os.path.join(args.std_folder,
                                            "stddevs.npy")).astype(np.float32)
        self.means = self.means[
            2, None, :,  # args.parameters.index(args.pred_type),
            :, :self.args.max_lat, :self.args.max_lon, ]
        self.stddevs = self.stddevs[
            2, None, :,  # args.parameters.index(args.pred_type),
            :, :self.args.max_lat, :self.args.max_lon, ]
        if args.dims == 2:
            self.stddevs = self.stddevs[:, args.plvl_used, :, :]
            self.means = self.means[:, args.plvl_used, :, :]
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
        data_x = reduce_sample_x(data_x, self.args, self.means, self.stddevs)

        if not self.infer and self.step == "train":

            data_y = np.load(self.datalist_y[idx])
            data_y = reduce_sample_y(data_y, self.args)

            return torch.from_numpy(data_x.copy()), torch.from_numpy(
                data_y.copy())
        else:

            if not self.infer and (self.step == "val" or self.step == "test"):
                data_y = np.load(self.datalist_y[idx])
                data_y = reduce_sample_y(data_y, self.args)

                return torch.from_numpy(data_x), torch.from_numpy(data_y)
            return torch.from_numpy(data_x)


class D500WeatherDataset(D500Dataset):

    def __init__(self,
                 sample_node: str,
                 label_node: str,
                 step="train",
                 infer=False,
                 year_dict=None,
                 args: UQDataClass = None):
        super().__init__()
        self.args = args
        self.input_node = sample_node
        self.label_node = label_node
        self.index = 0
        data_list = os.listdir(args.data_directory)
        pat_x = ["inputLST" + i for i in year_dict[step]]
        pat_y = ["inputLSTC" + i for i in year_dict[step]]
        self.datalist_x = [
            os.path.join(args.data_directory, i) for i in data_list
            if any(j in i for j in pat_x)
        ]
        self.datalist_y = [
            os.path.join(args.data_directory, i) for i in data_list
            if any(j in i for j in pat_y)
        ]
        self.means = np.load(os.path.join(args.std_folder,
                                          "means.npy")).astype(np.float32)
        self.stddevs = np.load(os.path.join(args.std_folder,
                                            "stddevs.npy")).astype(np.float32)
        self.means = self.means[
            2, None, :,  # args.parameters.index(args.pred_type),
            :, :self.args.max_lat, :self.args.max_lon, ]
        self.stddevs = self.stddevs[
            2, None, :,  # args.parameters.index(args.pred_type),
            :, :self.args.max_lat, :self.args.max_lon, ]
        if args.dims == 2:
            self.stddevs = self.stddevs[:, args.plvl_used, :, :]
            self.means = self.means[:, args.plvl_used, :, :]
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
        data_x = reduce_sample_x(data_x, self.args, self.means, self.stddevs)

        if not self.infer and self.step == "train":
            data_y = np.load(self.datalist_y[idx])
            data_y = reduce_sample_y(data_y, self.args)

            return data_x.copy(), data_y.copy()
        else:
            if not self.infer and (self.step == "val" or self.step == "test"):
                data_y = np.load(self.datalist_y[idx])
                data_y = reduce_sample_y(data_y, self.args)

                return data_x.copy(), data_y.copy()

            return data_x.copy()


class WeatherShuffleSampler(ShuffleSampler):

    def __init__(self,
                 dataset: Dataset,
                 batch_size: int,
                 seed: int = None,
                 drop_last_batch: bool = True,
                 events=None,
                 args: UQDataClass = None):

        super().__init__(dataset, batch_size, seed, drop_last_batch, events)
        self.batch_idx = 0
        self.args = args

    def __next__(self):

        if (self.drop_last_batch
                and self.batch_idx + self.batch_size > len(self.dataset)):
            raise StopIteration

        if self.batch_idx >= len(self.dataset):
            raise StopIteration

        batch_idx = self.sample_pool[
            self.batch_idx:min(self.batch_idx +
                               self.batch_size, len(self.dataset))]
        batch_data = []
        batch_label = []
        for idx in batch_idx:
            curr_data = self.dataset[idx]
            data_x = curr_data[0]
            data_y = curr_data[1]

            if self.args is not None:
                for aug in self.args.augmentation:  # Apply transformations if chosen
                    data_x, data_y = TRANSFORMATION_DICTIONARY[aug](data_x, data_y,
                                                                self.args)

            batch_data.append(data_x)
            batch_label.append(data_y)

        batch = {'input': np.array(batch_data), 'label': np.array(batch_label)}

        self.batch_idx += self.batch_size
        return batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def reset(self):
        super().reset()
        self.batch_idx = 0


class CallableD500WeatherDataset:

    def __init__(self, args: UQDataClass = None):

        self.args = args

        YEARS = [str(i) for i in range(1999, 2018)]
        year_dict = {}
        if args.fix_split:
            year_dict["test"] = ['2016', '2017']
            year_dict["val"] = ['2014', '2015']
            year_dict["train"] = [str(i) for i in range(1999, 2014)]
        else:
            random.shuffle(YEARS)
            year_dict["test"] = [YEARS[0], YEARS[-1]]
            year_dict["val"] = [YEARS[1], YEARS[-2]]
            year_dict["train"] = [
                i for e, i in enumerate(YEARS) if 1 < e < (len(YEARS) - 1)
            ]

        self.train_set = D500WeatherDataset(
            args=self.args,
            sample_node="input",
            label_node="label",
            step="train",
            year_dict=year_dict,
        )
        self.test_set = D500WeatherDataset(
            args=self.args,
            sample_node="input",
            label_node="label",
            step="test",
            year_dict=year_dict,
        )
        self.loss = torch.nn.L1Loss()
        self.input_node = "input"
        self.shape = [42, 352, 704]

    def __call__(self, *args, **kwargs):
        return self.train_set, self.test_set

    def __len__(self):
        return len(self.train_set)
