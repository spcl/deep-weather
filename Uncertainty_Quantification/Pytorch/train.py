from argparse import ArgumentParser
import loader
import random
from models import unet3d
from pytorch_lightning import Trainer

def main():

    # Set reproducability seed
    random.seed(42)

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default="3DUNet")

    # To pull the model name
    temp_args, _ = parser.parse_known_args()    

    parser.add_argument("--data_directory", type=str, default="~/RSTA_DATA")
    parser.add_argument("--parameters",nargs="+",default=["Temperature","SpecHum","VertVel","U","V","Geopotential","Divergence"])
    parser.add_argument("--augmentation", nargs="+", default=["RandomCrop","RandomHorizontalFlip","RandomVerticalFlip","Transpose"])
    parser.add_argument("--pressure_levels", nargs="+", default=[500,850])
    parser.add_argument("--time_steps",nargs="+",default=[0,24,48])
    parser.add_argument("--crop_x", type=int,default=256)
    parser.add_argument("--crop_y", type=int,default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--checkpoint_start", type=str, default=None)
    parser.add_argument("--grad_accumulation", type=int, default=1)

    #if temp_args.model_name == "3DUNet":
    #    parser = 3dunet.add_model_specific_args(parser)

    args = parser.parse_args()

    if args.model_name == "3DUNet":
        model = unet3d(
            sample_nr=len(loader.WeatherDataset(args, step="train")) // args.batch_size,
            base_lr=args.base_lr,
            max_lr=args.max_lr,
            in_channels=len(args.parameters)*len(args.time_steps),
            out_channels=1 # output temperature only
        )

    trainer = Trainer(gpus=args.gpus, accelerator='dp',accumulate_grad_batches=args.grad_accumulation, resume_from_checkpoint=args.checkpoint_start).from_argparse_args(args)

    dm = loader.WDatamodule(args)
    dm.setup(args)

    trainer.fit(model, dm.train, dm.val)

    result = trainer.test(test_dataloaders=dm.test)
    print(result)

if __name__ == "__main__":
    main()