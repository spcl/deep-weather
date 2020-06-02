import os



GRIB_PATH = '/mnt/data/weather/ETH_data/'
NPY_PATH = GRIB_PATH+'Preprocessed/'      # Numpy data path
TFR_PATH = NPY_PATH+'tfr/'                              # TensorFlow Record data path
CRPS_TEMP_PATH = '/mnt/data/weather/chyao/bias_corr_temp_ens5/'
CRPS_GEOP_PATH = '/mnt/data/weather/chyao/bias_corr_geo_ens5/'
if not os.path.exists(NPY_PATH):
    os.makedirs(NPY_PATH)
if not os.path.exists(TFR_PATH):
    os.makedirs(TFR_PATH)

USE_PRECOMPUTED_STDIZATION = True # Use the precomputed standardization files 'means.npy' and 'stddev.npy' for standardization
CRPS_PREPROCESSING = False        # Set to True if preprocessing data for CRPS prediction
CRPS_TEMP = True                  # To preprocess CRPS files for Temperature set to True, for Geopotential set to False. If not performing CRPS preprocessing, both will be transformed to tfr and this variable will be ignored
