import os
from argparse import ArgumentParser
import loader
import random
import torch
from models import unet3d, resnet2d_simple
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import numpy as np

YEARS = [str(i) for i in range(1999, 2018)]


def main():

    # Set reproducability seed
    random.seed(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet2d_simple",
        help="{3DUNet,resnet2d_simple}",
    )

    # To pull the model name
    temp_args, _ = parser.parse_known_args()

    parser.add_argument(
        "--data_directory", type=str, default="/users/petergro/RSTA_DATA"
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
        help='["RandomCrop","RandomHorizontalFlip","RandomVerticalFlip","Transpose"]',
    )
    parser.add_argument(
        "--max_lat",
        type=int,
        default=352,
        help="Maximum latitude used as crop limit to allow for down and upscaling",
    )
    parser.add_argument(
        "--max_lon",
        type=int,
        default=702,
        help="Maximum longitude used as crop limit to allow for down and upscaling",
    )
    parser.add_argument("--pressure_levels", nargs="+", default=[500, 850])
    parser.add_argument("--dims", type=int, default=2)
    parser.add_argument("--plvl_used", nargs="+", default=1)
    parser.add_argument("--time_steps", nargs="+", default=[0, 24, 48])
    parser.add_argument("--perturbations", nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--crop_lon", type=int, default=256)
    parser.add_argument("--crop_lat", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--checkpoint_start", type=str, default=None)
    parser.add_argument("--pred_type", type=str, default="Temperature")
    parser.add_argument("--aggr_type", type=str, default="Spread", help="Spread | Mean")
    parser.add_argument(
        "--std_folder",
        type=str,
        default="/users/petergro/std",
        help="Folder with the means.npy and stddevs.npy files generated in the data generation",
    )
    parser.add_argument("--output", type=str, default="./output")

    # if temp_args.model_name == "3DUNet":
    #    parser = 3dunet.add_model_specific_args(parser)

    args = parser.parse_args()

    # Set the test set to be the last two years
    year_dict = {}
    year_dict["test"] = [YEARS[0], YEARS[-1]]
    year_dict["val"] = [YEARS[1], YEARS[-2]]
    year_dict["train"] = [i for e, i in enumerate(YEARS) if 1 < e < (len(YEARS) - 1)]

    loaded_data = loader.WeatherDataset(
        args, step="test", infer=True, year_dict=year_dict
    )

    if args.model_name == "3DUNet":
        model = unet3d(
            sample_nr=1,
            base_lr=1,
            max_lr=1,
            in_channels=len(args.parameters)
            * len(args.time_steps)
            * 2,  # 2 because we use either the mean or the std + the unperturbed trajectory as input
            out_channels=1,  # output temperature only
        )
    elif args.model_name == "resnet2d_simple":
        model = resnet2d_simple(
            sample_nr=1,
            base_lr=1,
            max_lr=1,
            in_channels=len(args.parameters)
            * len(args.time_steps)
            * 2,  # 2 because we use either the mean or the std + the unperturbed trajectory as input
            out_channels=1,  # output temperature only
        )

    checkpoint = torch.load(
        args.checkpoint_start, map_location=lambda storage, loc: storage
    )
    model.load_state_dict(checkpoint["state_dict"])
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    model.eval()
    model = model.to("cpu") if args.gpus is None else model.to("cuda")
    # --min_epochs 1 --max_epochs 30 --gpus 2 --accelerator ddp --accumulate_grad_batches 1 --resume_from_checkpoint None
    results = []
    for e, i in enumerate(loaded_data):
        pred = (
            model(i.unsqueeze(0))
            if args.gpus is None
            else model(i.unsqueeze(0).to("cuda"))
        )
        # file_name = args.output + "/sample_" + str(e) + ".npy"
        results.append(pred.cpu().detach().numpy())
    np.save(args.output + "/inferred_result.npy", np.array(results))


if __name__ == "__main__":
    main()
