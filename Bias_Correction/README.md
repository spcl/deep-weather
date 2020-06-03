## A. Preprocessing pipeline
1. In file bias-correction/data/GRIBglobal.py, in line 272-276, change the path variables. Keep tfrecord_output_path to be bias-correction/data//tfdata (since this is the default path for the models to read the data)
1*. The default setting is to get the mean of the 10 ensembles. To change to 5 ensembles or other custom operations, the change can be made in the meanparse function.
2. Run python GRIBglobal.py. The time for processing is expected to be very long 
3. copy files mean*.npy and std*.npy from npy_output_path to tfrecord_output_path 

## B. Running the model
1. In bias-correction/config.py, set variable temp/geo to True if 850hPa is used and to False if 500hPa is used.
2. Run commands to train the model, some examples of the commands are listed below. For details of the arguments, please check args.py
3. check the trained models in bias-correction/ckpt/<model>/<exp> 

UNET0:
python cross_validate.py unet3 --unet_levels=0 --epoch=40 --nfilters=32 --val_interval=2 --save_interval=10 --lr=5e-4  --plot=False --batch_size=2 --exp=formal_l0_32_geo --crop=False --crop_stack=False --recompute=0 | tee unet_l0_32

UNET1+LCN+reg: 
python cross_validate.py unet3_local --unet_levels=1 --epoch=40 --nfilters=32 --lcn_kernel=1,1,1 --val_interval=2 --save_interval=10 --lr=5e-4  --plot=False --batch_size=1 --exp=formal_local_l1_32_temp_ens5 --crop=False --crop_stack=False --recompute=0 --regularize=True --alpha=1e1 | tee unet_local_l1_32_reg_ens5_temp_ref

UNET2+LCN no reg: 
python cross_validate.py unet3_local --unet_levels=2 --epoch=40 --nfilters=32 --lcn_kernel=1,1,1 --val_interval=2 --save_interval=10 --lr=5e-4  --plot=False --batch_size=1 --exp=formal_local_l1_32_temp_ens5 --crop=False --crop_stack=False --recompute=0 --regularize=False --alpha=1e1 | tee unet_local_l1_32_temp_ref

## C. Load the model and get predictions
1. Set the flag "resume_iter" to be the epoch run with the saved model. To use this method to load the model, experiment name, model needs to be the same as the training command.
2. main.py script default dumps out predictions on the daily basis (file names are the dates) to bias-correction/log/<model>/<exp> in npy format. To customize the function to dump out predictions of different years, main.py file need to be changed. 

Resume at the 40 epoch of UNET1+LCN+reg:
python main.py unet3_local --unet_levels=1 --epoch=40 --nfilters=32 --lcn_kernel=1,1,1 --val_interval=2 --save_interval=20 --lr=5e-4  --plot=False --batch_size=1 --exp=formal_local_l1_32_temp_ens5 --crop=False --crop_stack=False --resume_iter=40
