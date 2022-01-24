""" A recipe to reproduce the UQ of the following repo:
    https://github.com/spcl/deep-weather

"""
import math

import deep500 as d5
# Using PyTorch as the framework
import deep500.frameworks.pytorch as d5fw

import torch
from events import *
from models import *
from utils import *
from losses import *
import loader
from scheduler import CyclicLRScheduler

args = args_parser()


def model(batch_size, in_channels, out_channels, classes=None, shape=None):
    net = resnet2d_simple(
        in_channels=in_channels,
        out_channels=out_channels,
    )
    return d5fw.PyTorchNativeNetwork(net), "input", "output"


def executor():
    mod, _, _ = model(
        None,
        in_channels=len(args.parameters) * len(args.time_steps) * 2,
        out_channels=1,
        classes=None,
        shape=None,
    )
    loss_op = MixLoss(torch.nn.L1Loss(), MSSSIMLoss(), 0.84)
    return d5fw.PyTorchNativeGraphExecutor(mod.module, loss_op, "input")


# Fixed Components
FIXED = {
    "model": model,
    "model_kwargs": {
        "in_channels": len(args.parameters) * len(args.time_steps) * 2,
        "out_channels": 1,
    },
    "dataset": loader.CallableD500WeatherDataset(args),
    "epochs": args.epochs,
}

# Mutable Components
MUTABLE = {
    "batch_size": args.batch_size,
    "executor": executor(),
    "executor_kwargs": dict(device=d5.GPUDevice()),
    "train_sampler": loader.WeatherShuffleSampler,
    "train_sampler_kwargs": dict(seed=args.seed),
    "validation_sampler": loader.WeatherShuffleSampler,
    "validation_sampler_kwargs": dict(seed=args.seed),
    "optimizer": d5fw.AdamOptimizer,
    "optimizer_kwargs": dict(learning_rate=1e-2),
    "events": [
        RMSETerminalBarEvent(loader.CallableD500WeatherDataset(args).test_set,
                             loader.WeatherShuffleSampler,
                             batch_size=args.batch_size),
        CyclicLRScheduler(
            per_epoch=True,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            step_size_up=(len(loader.CallableD500WeatherDataset(args)) //
                          args.batch_size) // 2,
            step_size_down=None,
            mode="triangular2",
            gamma=1.0,
            scale_fn=None,
            scale_mode="cycle",
            cycle_momentum=False,
        ),
    ],
}

# Acceptable Metrics
METRICS = []

if __name__ == "__main__":
    d5.run_recipe(FIXED, MUTABLE, METRICS) or exit(1)
