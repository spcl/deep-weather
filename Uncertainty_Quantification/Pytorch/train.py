from argparse import ArgumentParser
import loader
import random
from models import unet3d
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

YEARS = [str(i) for i in range(1999, 2018)]


def main():

    # Set reproducability seed
    random.seed(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default="3DUNet")

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
            "RandomCrop",
            "RandomHorizontalFlip",
            "RandomVerticalFlip",
            "Transpose",
        ],
    )
    parser.add_argument("--pressure_levels", nargs="+", default=[500, 850])
    parser.add_argument("--plvl_used", type=int, default=1)
    parser.add_argument("--time_steps", nargs="+", default=[0, 24, 48])
    parser.add_argument("--perturbations", nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--crop_lon", type=int, default=256)
    parser.add_argument("--crop_lat", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=20)
    parser.add_argument("--checkpoint_start", type=str, default=None)
    parser.add_argument("--grad_accumulation", type=int, default=1)
    parser.add_argument("--pred_type", type=str, default="Temperature")
    parser.add_argument("--aggr_type", type=str, default="Mean")
    parser.add_argument(
        "--fix_split",
        type=bool,
        default=True,
        help="weather the train, val, test split are random or fixed",
    )

    # if temp_args.model_name == "3DUNet":
    #    parser = 3dunet.add_model_specific_args(parser)

    args = parser.parse_args()

    # Adapt the sample split
    year_dict = {}
    if args.fix_split:
        year_dict["test"] = [YEARS[0], YEARS[-1]]
        year_dict["val"] = [YEARS[1], YEARS[-2]]
        year_dict["train"] = [
            i for e, i in enumerate(YEARS) if 1 < e < (len(YEARS) - 1)
        ]
    else:
        random.shuffle(YEARS)
        year_dict["test"] = [YEARS[0], YEARS[-1]]
        year_dict["val"] = [YEARS[1], YEARS[-2]]
        year_dict["train"] = [
            i for e, i in enumerate(YEARS) if 1 < e < (len(YEARS) - 1)
        ]

    if args.model_name == "3DUNet":
        model = unet3d(
            sample_nr=len(
                loader.WeatherDataset(args, step="train", year_dict=year_dict)
            )
            // args.batch_size,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            in_channels=len(args.parameters)
            * len(args.time_steps)
            * len(args.perturbations),
            out_channels=1,  # output temperature only
        )

    trainer = Trainer(
        gpus=args.gpus,
        accelerator="ddp",
        distributed_backend="ddp",
        accumulate_grad_batches=args.grad_accumulation,
        resume_from_checkpoint=args.checkpoint_start,
        callbacks=[EarlyStopping(monitor="val_loss")],
    )  # .from_argparse_args(args)

    dm = loader.WDatamodule(args, year_dict=year_dict)
    dm.setup(args)

    trainer.fit(model, dm.train, dm.val)

    result = trainer.test(test_dataloaders=dm.test)
    print(result)


if __name__ == "__main__":
    main()
