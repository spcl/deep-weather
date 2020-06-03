# Deep Learning for Post-Processing Ensemble Weather Forecasts 
 
We make available the data as well as the code that is necessary to run the models in our [paper](https://arxiv.org/abs/2005.08748) through this repository. It is our hope that our findings and the data can be used to further advance weather forecasting research.

## Research description

Our research focuses on applying recent architectures from Deep Learning to Ensemble Weather Forecasts. To achieve this we use global reforecast data from the ECMWF that we call **ENS10**, as well as reanalysis data **ERA5**. More specifically, ENS10 is aimed at providing researchers with a basic dataset of forecasted values that would be used in modern numerical weather prediction pipelines. Using ERA5 data as ground truth, we then use a subset of the ensemble forecasts to post-process and improve. Our aim is to help weather forecasting centers predict extreme weather events cheaper and more accurately. 

<p align="center">
<img width="40%" src="/report/G_Winston_E10_step1.png">
<img width="40%" src="/report/G_Winston_B5U5C-E10_step1.png">
</p>

In the case of tropical cyclone [Winston](https://en.wikipedia.org/wiki/Cyclone_Winston) we achieve a relative improvement of over 26% in forecast skill, measured in Continuous Ranked Probability Score (CRPS) over the full 10 member ensemble, using a subset of five trajectories. Additionally, the models specifically predict the future path of the cyclone more accurately. 

## Dependencies
In order to run our code in Python 3 through a virtual environment: Clone this repository, open a terminal, set the working directory to this directory and run:
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Data
Our raw GRIB data is available under the following links:
- [ENS10](https://confluence.ecmwf.int/display/UDOC/ECMWF+ENS+for+Machine+Learning+%28ENS4ML%29+Dataset)
- [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form)

To transform the data into Numpy arrays and TensorFlow records, refer to our preprocessing steps.

## Contents

We separate our experiments into the exploration of ensemble output bias correction, which needs ERA5 data as the ground truth, and ensemble uncertainty forecasting using reduced ensemble members, which runs solely on ENS10 data.
To make a combined prediction aimed at reducing the forecast skill (CRPS), set the appropriate flags in [parameters.py](Uncertainty_Quantification\parameters.py) additionally, provide a path to the bias corrected mean and ground truth in the appropriate form in [preprocessing_parameters.py](Uncertainty_Quantification\PreProcessing\preprocessing_parameters.py).

### Contents structure
`Bias_Correction` Preprocessing and Model for Bias Correction, refer to `Bias_Correction/README.md`
`Uncertainty_Quantification` Preprocessing and Model for Uncertainty Quantification
 - `\parameters.py` All hyperparameters and settings for the model, edit this file to add the paths to your preprocessed data
 - `\predict.py` Evaluates the model on test data
 - `\RESNET2D.py` Our TensorFlow model for Uncertainty Quantification
 - `\train.py` Runs training and validation operations on the model
 - `\Preprocessing` Contains all necessary files to transform GRIB files into TensorFlow records
    - `\GRIB2npy.py` Converts GRIB files to Numpy arrays, needs to be adjusted for the selected parameters
    - `\npy2tfr.py` Converts the Numpy arrays to TensorFlow records and performs the Local Area-wise Standardization (LAS) described in our paper
    - `\preprocessing_parameters` Parameters for all preprocessing steps, be sure to edit the path folders
    - `\means.npy` Precomputed means for the LAS
    - `\stddevs.npy` Precomputed standard deviations for the LAS

## How to cite
```
@article{grnquist2020deep,
    title={Deep Learning for Post-Processing Ensemble Weather Forecasts},
    author={Peter Grönquist and Chengyuan Yao and Tal Ben-Nun and Nikoli Dryden and Peter Dueben and Shigang Li and Torsten Hoefler},
    year={2020},
    eprint={2005.08748},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Should there be any questions, feel free to contact us.


