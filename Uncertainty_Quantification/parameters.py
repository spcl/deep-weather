#!/usr/bin/env python3

import os
import random

# Hyperparameter file, edit this file to suit your needs

LEARNING_RATE = 0.001   # Model learning rate
EPOCHS = 500            # Nr of epochs
PATIENCE = 100          # Nr of epochs to wait without progress before early stopping
PSAVE = 100             # Periodically save model at PSAVE nr of epochs
BATCH_SIZE = 2          # Batch - VRAM limited

FOLDER = ''
#GRIB_DATA_DIRECTORY = "path/to/your/grib files"
NPY_DATA_DIRECTORY = "path/to/your/npy preprocessed files"+FOLDER
SAVE_PATH = "./tf/"
LOGS_PATH = "./tb/"
LOAD_MODEL = False      # Start training from existing model
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)
MODEL_NAME = 'model'  # Model name to load, 'model' by default

PREDICT_TEMP = True # Set to False for Geopotential
CONCAT_NWP = True   # Concatenate NWP to output features before loss, ensures additional stability. An alternative would be to use attention
DO_CRPS = False        # Train on CRPS

NR_TIMESTEPS = 3    # t, t+24h, t+48h
NR_CHANNELS = 14    # 7 new params trajectory 0 + 7 params stddev of used trajectories
INPUT_SIZEL = 720   # Input feature Length
INPUT_SIZEW = 361   # Input feature Width
INPUT_DEPTH = 2     # Input depth
OUTPUT_SIZEL = 720  # Output feature Length
OUTPUT_SIZEW = 361  # Output feature Width
OUTPUT_DEPTH = 2    # Depth of output
OFF_IMAGE_FILL = 0  # What to fill an image with if padding is required to make Tensor
OFF_LABEL_FILL = 0  # What to fill a label with if padding is required to make Tensor
BASEFILTER_SIZE = 16# Nr of filters in the first layer, doubled at each layer after that

random_shuffle = True
DATE_TO_PREDICT = 2017  # Year to predict in the test set

PATHT = NPY_DATA_DIRECTORY+"/train/"     # Training dataset path
PATHV = NPY_DATA_DIRECTORY+"/val/"       # Validation dataset path
PATHTE = NPY_DATA_DIRECTORY+"/test/"     # Test dataset path

TRAINING_LIST = os.listdir(PATHT)
VALIDATION_LIST = os.listdir(PATHV)
TEST_LIST = os.listdir(PATHTE)
if random_shuffle:
    random.shuffle(TRAINING_LIST)        # Shuffle training files
    random.shuffle(VALIDATION_LIST)      # Shuffle validation files
FILE_NAMES = [PATHT+x for x in TRAINING_LIST]
FILE_NAMESV = [PATHV+x for x in VALIDATION_LIST]
FILE_NAMEST = [PATHTE+x for x in TEST_LIST]



