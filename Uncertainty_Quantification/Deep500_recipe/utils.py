import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from typing import Any, List, Optional

from dataclasses import dataclass, field


@dataclass()
class UQDataClass:

    model_name: str = field(default="resnet2d_simple",
                            metadata={'help': "{3DUNet, resnet2d_simple}"})

    seed: int = field(default=42,
                      metadata={'help': "Random Seed (default: 42)"})

    epochs: int = field(default=1,
                        metadata={'help': "Number of Epochs (default: 1)"})

    data_directory: str = field(
        default="/users/petergro/RSTA_DATA",
        metadata={'help': 'Pass your data directoy here'})
    std_folder: str = field(
        default="/users/petergro/std",
        metadata={
            'help':
            'Folder with the means.npy and stddevs.npy files generated in the data generation'
        })

    parameters: List[str] = field(
        default_factory=lambda: [
            "Temperature",
            "SpecHum",
            "VertVel",
            "U",
            "V",
            "Geopotential",
            "Divergence",
        ],
        metadata={'help': 'The parameters that will be used'})

    augmentation: List[str] = field(
        default_factory=lambda: [],
        metadata={
            'help':
            '["RandomCrop","RandomHorizontalFlip","RandomVerticalFlip","Transpose"]'
        })

    max_lat: int = field(
        default=352,
        metadata={
            'help':
            'Maximum latitude used as crop limit to allow for down and upscaling'
        })

    max_lon: int = field(
        default=704,
        metadata={
            'help':
            'Maximum longitude used as crop limit to allow for down and upscaling'
        })

    pressure_levels: List[int] = field(
        default_factory=lambda: [500, 850],
        metadata={'help': 'What pressure levels are available'})

    dims: int = field(
        default=2,
        metadata={'help': 'How many dimensions are we predicting in, 2 or 3'})

    plvl_used: int = field(
        default=1,
        metadata={
            'help':
            'if --dims is 2, which pressure level are we predicting (index)'
        })

    time_steps: List[int] = field(
        default_factory=lambda: [0, 24, 48],
        metadata={'help': 'List of timesteps that are available to use'})

    perturbations: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4],
        metadata={
            'help': 'What are the perturbation numbers that we are using'
        })

    crop_lon: int = field(
        default=256,
        metadata={
            'help':
            'What is the crop size in longitude points for the random crop'
        })

    crop_lat: int = field(
        default=256,
        metadata={
            'help':
            'What is the crop size in latitude points for the random crop'
        })

    batch_size: int = field(default=2, metadata={'help': 'The batch size'})

    base_lr: float = field(
        default=1e-6,
        metadata={
            'help':
            'Base learning rate for the cyclical learning rate scheduler'
        })

    max_lr: float = field(
        default=1e-2,
        metadata={
            'help':
            'Maximum learning rate for the cyclical learning rate scheduler'
        })

    num_workers: int = field(
        default=8, metadata={'help': 'Set the number of dataloader workers'})

    pred_type: str = field(
        default='Temperature',
        metadata={'help': 'Which parameter is being predicted'})

    aggr_type: str = field(default='Spread',
                           metadata={'help': 'Spread | Mean'})

    fix_split: bool = field(
        default=True,
        metadata={
            'help': 'Whether the train, val, test split are random or fixed'
        })


def args_parser():

    parser = ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet2d_simple",
        help="{3DUNet,resnet2d_simple}",
    )

    # To pull the model name
    temp_args, _ = parser.parse_known_args()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed (default: 42)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of Epochs (default: 1)",
    )

    parser.add_argument(
        "--data_directory",
        type=str,
        default="/users/petergro/RSTA_DATA",
        help="Pass your data directoy here",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        default=[
            "Temperature",
            "SpecHum",
            "VertVel",
            "U",
            "V",
            "Geopotential",
            "Divergence",
        ],
        help="The parameters that will be used",
    )
    parser.add_argument(
        "--augmentation",
        nargs="+",
        default=[
            # "RandomCrop",
            # "RandomHorizontalFlip",
            # "RandomVerticalFlip",
            # "Transpose",
        ],
        help=
        '["RandomCrop","RandomHorizontalFlip","RandomVerticalFlip","Transpose"]',
    )
    parser.add_argument(
        "--max_lat",
        type=int,
        default=352,
        help=
        "Maximum latitude used as crop limit to allow for down and upscaling",
    )
    parser.add_argument(
        "--max_lon",
        type=int,
        default=704,
        help=
        "Maximum longitude used as crop limit to allow for down and upscaling",
    )
    parser.add_argument(
        "--pressure_levels",
        nargs="+",
        default=[500, 850],
        help="What pressure levels are available",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2,
        help="How many dimensions are we predicting in, 2 or 3",
    )
    parser.add_argument(
        "--plvl_used",
        nargs="+",
        default=1,
        help="if --dims is 2, which pressure level are we predicting (index)",
    )
    parser.add_argument(
        "--time_steps",
        nargs="+",
        default=[0, 24, 48],
        help="List of timesteps that are available to use",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="What are the perturbation numbers that we are using",
    )
    parser.add_argument(
        "--crop_lon",
        type=int,
        default=256,
        help="What is the crop size in longitude points for the random crop",
    )
    parser.add_argument(
        "--crop_lat",
        type=int,
        default=256,
        help="What is the crop size in latitude points for the random crop",
    )
    parser.add_argument("--batch_size",
                        type=int,
                        default=8,
                        help="The batch size")
    # todo: Add LR scheduler!
    parser.add_argument(
        "--base_lr",
        type=float,
        default=1e-6,
        help="Base learning rate for the cyclical learning rate scheduler",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=1e-2,
        help="Maximum learning rate for the cyclical learning rate scheduler",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Set the number of dataloader workers",
    )
    # TODO: gradient accumulator
    # parser.add_argument(
    #     "--grad_accumulation",
    #     type=int,
    #     default=1,
    #     help="How many gradient passes should be accumulated",
    # )
    parser.add_argument(
        "--pred_type",
        type=str,
        default="Temperature",
        help="Which parameter is being predicted",
    )
    parser.add_argument("--aggr_type",
                        type=str,
                        default="Spread",
                        help="Spread | Mean")
    parser.add_argument(
        "--std_folder",
        type=str,
        default="/users/petergro/std",
        help=
        "Folder with the means.npy and stddevs.npy files generated in the data generation",
    )
    parser.add_argument(
        "--fix_split",
        type=bool,
        default=True,
        help="Whether the train, val, test split are random or fixed",
    )

    # if temp_args.model_name == "3DUNet":
    #    parser = 3dunet.add_model_specific_args(parser)

    args = parser.parse_args()

    return UQDataClass(**vars(args))


def reduce_sample_x(
    data_x,
    args,
    means,
    stddevs,
):
    """Crops longitude, latitude, aggregates trajectories into mean and stddev, reshapes and standardizes the input data

    Args:
        data_x (np.array): input data
        args (Object): Parsed argumetns
        means (np.array): Means from LAS
        stddevs (np.array): Stddevs from LAS

    Returns:
        np.array: reduced input data
    """
    # For now plvl used only works with 2d data, can be scaled to be able to select 3d data later if needed
    # crop to work with 5 pooling operations
    data_x = data_x[:, :, :, :, :args.max_lat, :args.max_lon]
    op = np.mean if args.aggr_type == "Mean" else np.std
    stdized = (data_x[:, 0, None, :, :, :, :] - means) / stddevs
    data_x = np.concatenate([op(data_x, axis=1, keepdims=True), stdized],
                            axis=1)
    if args.dims == 2:
        data_x = data_x[:, :, :, args.plvl_used, :, :]
        data_x = np.reshape(
            data_x,
            [
                data_x.shape[0] * data_x.shape[1] * data_x.shape[2],
                data_x.shape[3],
                data_x.shape[4],
            ],
        )
    else:
        data_x = np.reshape(
            data_x,
            [
                data_x.shape[0] * data_x.shape[1] * data_x.shape[2],
                data_x.shape[3],
                data_x.shape[4],
                data_x.shape[5],
            ],
        )
    return data_x


def reduce_sample_y(data_y, args):
    """Crops longitude, latitude, selects the prediction parameter and pressure level for the ground truth.

    Args:
        data_y (np.array): Ground truth sample
        args (Object): Parsed arguments

    Returns:
        np.array: reduced ground truth
    """
    # crop to work with 5 pooling operations
    data_y = data_y[:, :, :, :args.max_lat, :args.max_lon]
    ind = 1 if args.aggr_type == "Mean" else 0
    if args.dims == 2:
        data_y = data_y[ind, None,
                        args.parameters.index(args.pred_type),
                        args.plvl_used, :, :]
    else:
        data_y = data_y[ind, None,
                        args.parameters.index(args.pred_type), :, :, :]
    return data_y


def random_crop(data_x, data_y, args):
    """Randomly crops both input and ground truth data to window size defined in args

    Args:
        data_x (np.array): input data
        data_y (np.array): ground truth data
        args (Object): parsed arguments

    Returns:
        (np.array, np.array): cropped data
    """
    max_lat = data_y.shape[-2] - args.crop_lat
    max_lon = data_y.shape[-1] - args.crop_lon
    lat = random.randint(0, max_lat)
    lon = random.randint(0, max_lon)
    if args.dims == 2:
        data_x = data_x[:, lat:lat + args.crop_lat, lon:lon + args.crop_lon]
        data_y = data_y[:, lat:lat + args.crop_lat, lon:lon + args.crop_lon]
    else:
        data_x = data_x[:, :, lat:lat + args.crop_lat, lon:lon + args.crop_lon]
        data_y = data_y[:, :, lat:lat + args.crop_lat, lon:lon + args.crop_lon]
    return data_x, data_y


def horizontal_flip(data_x, data_y, args):
    """Randomly performs a horizontal flip on both input and ground truth data

    Args:
        data_x (np.array): input data
        data_y (np.array): ground truth data
        args (Object): parsed arguments

    Returns:
        (np.array, np.array): flipped data
    """
    if random.random() < 0.5:
        data_x = np.flip(data_x, -1)
        data_y = np.flip(data_y, -1)
    return data_x, data_y


def vertical_flip(data_x, data_y, args):
    """Randomly performs a vertical flip on both input and ground truth data

    Args:
        data_x (np.array): input data
        data_y (np.array): ground truth data
        args (Object): parsed arguments

    Returns:
        (np.array, np.array): flipped data
    """
    if random.random() < 0.5:
        data_x = np.flip(data_x, -2)
        data_y = np.flip(data_y, -2)
    return data_x, data_y


def transpose(data_x, data_y, args):
    """Randomly transposes both input and ground truth data

    Args:
        data_x (np.array): input data
        data_y (np.array): ground truth data
        args (Object): parsed arguments

    Returns:
        (np.array, np.array): transposed data
    """
    if random.random() < 0.5:
        if args.dims == 2:
            data_x = data_x.transpose(0, 2, 1)
            data_y = data_y.transpose(0, 2, 1)
        else:
            data_x = data_x.transpose(0, 1, 3, 2)
            data_y = data_y.transpose(0, 1, 3, 2)
    return data_x, data_y


def standardize(x, means, stddevs):
    return (x - means) / stddevs
