# Uncertainty Quantification with PyTorch-Lightning

PyTorch-Lightning is a powerful tool that allows for mixed-precision data-parallel training, with automatic hyperparameter tuning.
Therefore, our latest models and data augmentation steps are made accessible in this framework.

## Getting Started

To run the PyTorch models and add your own, first be sure to use the preprocessing code to generate Numpy files from the raw GRIB files.
You do not need the TFRecord files for this framework.

### Structure

- `\infer.py` Runs inference on our models
- `\loader.py` Dataloader definition for our custom datasets
- `\Model_Filter.ipynb` Notebook to visualize weights and layers
- `\models.py` Our PyTorch models
- `\ssim.py` SSIM and MS-SSIM losses
- `\train.py` Runs the training on our models
- `\utils.py` Utility functions for dataloading and augmentation
- `\pytorch_environment.yml` Our conda environment file

### Prerequisites

To run this code you will have to install the conda or miniconda package manager.

### Installing

If you have conda installed, but have not set up an environment yet, install it with the following line:

```
conda install -f pytorch_environment.yml
```
and activate the environment with
```
conda activate Weather
```


If you already have a conda environment installed for this project, activate it and update it with the following line:
```
conda env update --file pytorch_environment.yml
```

Once you have your conda environment set up, be sure that you have all the Numpy files generated and ready in the same data folder.


## Training and Inference

To find the full list of command line arguments, refer to the [documentation](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api) or run
```
python train.py -h
```
for training and
```
python infer.py -h
```
for inference.

### Train the models

To reproduce our results run

```
python train.py --min_epochs 1 --max_epochs 10 --accumulate_grad_batches 1 --model_name resnet2d_simple --gpus 1 --batch_size 2 --data_directory YOUR/DIRECTORY
```
Per default, the number of pressure levels being used is 1, and therefore, the model is 2d. If you want to switch to using more pressure levels as input, for example with our 3D Unet, set
```
--dims 3 --model_name 3DUnet
```
this will change the network architecture to be three-dimensional. 
When training, the data is also automatically cropped to a multiple of 2 to the power of 5, to account for the downpooling and upsampling operations in the 3D Unet. If you only use the 2d resnet, or your own model without downsampling, adapt the `max_lat` and `max_lon` arguments accordingly.

If you need to resume training from a checkpoint, add the following argument with the path to your desired checkpoint
```
--resume_from_checkpoint lightning_logs/version_XXXX/checkpoints/epoch\=X.ckpt
```

At the end of the run, the model will evaluate itself on the test set and print the MSE loss values.

### Inference

To run inference on your test set, simply run the following

```
python infer.py --model_name resnet2d_simple --checkpoint_start lightning_logs/version_XXXX/checkpoints/epoch\=X.ckpt --gpus 1 --output ./YOUR_OUTPUT_FOLDER
```

