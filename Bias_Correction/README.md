# Output Bias Correction
CUDA device is required to reproduce the result. 
## A. Preprocessing pipeline (from GRIB format to TFRecord)
1. In file _Bias-Correction/data/GRIBglobal.py_, in line 282-286, change the path variables. It is recommended to keep _tfrecord_output_path_ to be _Bias-Correction/data/tfdata_ (since this is the default path for the models to read the data). If using other _tfrecord_output_path_, copying or soft link _Bias-Correction/data/tfdata_ is required. (Optional) The default setting is to get the mean of the 5 ensembles. To change to other number of ensembles, the change can be made in the meanparse function in _Bias-Correction/data/GRIBglobal.py_.

2. Run python script GRIBglobal.py. The time for processing is expected to be very long (can be around 16 hours).

3. copy files _mean*.npy_ and _std*.npy_ from _npy_output_path_ to _tfrecord_output_path_.


## B. Running the model
1. In _Bias-Correction/config.py_, set variable _temp/geo_ to True if training/testing on 850hPa temperature and to False if training/testing on 500hPa geopotential.

2. Run commands to train the model, some examples of the commands are listed below. For details of the arguments, please check _Bias-Correction/args.py_

3. check the trained models in _Bias-Correction/ckpt/[model]/[exp]_

4. For LCN with reg, alpha is the strength of the regularization. To reproduce the result in the paper, set alpha to be 1e1 (regularize needs to be True). 

```
UNET0:
python cross_validate.py unet3 --unet_levels=0 --epoch=40 --nfilters=32 --val_interval=2 --save_interval=10 --lr=5e-4  --plot=False --batch_size=2 --exp=formal_l0_32_geo --crop=False --crop_stack=False --recompute=0 | tee unet_l0_32

UNET1+LCN+reg:
python cross_validate.py unet3_local --unet_levels=1 --epoch=40 --nfilters=32 --lcn_kernel=1,1,1 --val_interval=2 --save_interval=10 --lr=5e-4  --plot=False --batch_size=1 --exp=formal_local_l1_32_temp_ens5 --crop=False --crop_stack=False --recompute=0 --regularize=True --alpha=1e1 | tee unet_local_l1_32_reg_ens5_temp_ref

UNET2+LCN no reg:
python cross_validate.py unet3_local --unet_levels=2 --epoch=40 --nfilters=32 --lcn_kernel=1,1,1 --val_interval=2 --save_interval=10 --lr=5e-4  --plot=False --batch_size=1 --exp=formal_local_l1_32_temp_ens5 --crop=False --crop_stack=False --recompute=0 --regularize=False --alpha=1e1 | tee unet_local_l1_32_temp_ref
```

## C. Load the model and get corrected mean predictions
1. Set the flag "resume_iter" to be the epoch run with the saved model. In order to use this method to load the model, experiment name, model needs to be the same as the training command.

2. _main.py_ script default dumps out predictions on the daily basis (file names are the dates) to _Bias-Correction/log/[model]/[exp]_ in npy format. The output is a 3 grids with shape [3, 361, 720]. The first grid is the prediction of mean, the second grid is the numerical forecast (mean of the ensemble members), and the third grid is ERA5 ground truth. To customize the function to dump out predictions of different years, in line 37 of _main.py_ file, change the _year_test_ to be the the list of years to be predicted.

3. The output of the corrected mean predictions can be used further for the CRPS training.

```
Resume at the 40 epoch of UNET1+LCN+reg:
python main.py unet3_local --unet_levels=1 --epoch=40 --nfilters=32 --lcn_kernel=1,1,1 --val_interval=2 --save_interval=20 --lr=5e-4  --plot=False --batch_size=1 --exp=formal_local_l1_32_temp_ens5 --crop=False --crop_stack=False --resume_iter=40
```
