# Deep500 Recipe for Uncertainty Quantification

[Deep500](https://www.deep500.org)  is A modular benchmarking infrastructure for 
high-performance deep learning — from the single operator 
to distributed training. In this directory, we  reproduce the result of
*uncertainty quantification* task using deep500 recipe.


## Usage

To run the code, you need to install [Deep500](https://github.com/deep500/deep500). Then, install  `pytorch_msssim` using the following line:

```
pip install pytorch_msssim
```


Now, you can train the model using the following:

```
python recipe_UQ.py  --data_directory YOUR/DATA/DIRECTORY   --std_folder YOUR/STD&MEAN_MAP/FOLDER  --batch_size 2 --epochs 6  --fix_split True
```


## Dataset

The data should be downloaded as stated [here](https://github.com/spcl/deep-weather#data). 
`mean.npy` and `stddevs.npy` is also needed (you can find them [here](https://github.com/spcl/deep-weather/tree/master/Uncertainty_Quantification/Preprocessing)).
The data has 5 dimensions with the following attributes:

- `Leadtime (3)`: The data for 0, 24h, and 48h
- `Type (7)`: Temperature, Geopotential, U component of wind, V component of wind, Divergence, Vertical velocity, and Specific humidity  
- `Pressure Level (2)`: 500 and 850hpa.
- `Latitude (361)`
- `Longitude (720)`