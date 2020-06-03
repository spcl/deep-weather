#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import random
import preprocessing_parameters
import os
from scipy.ndimage import gaussian_filter1d

# Edit this file to your needs
wsize = 7 # window size for localized standardization



def Npy2TFRecord(Array, ArrayC, writer): #, Size
    features = {
        'Array': tf.train.Feature(float_list=tf.train.FloatList(value=Array[:,:,:,:,:].flatten())),
        'ArrayC': tf.train.Feature(float_list=tf.train.FloatList(value=ArrayC[:,:,:,:,:].flatten()))
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(example.SerializeToString())

PATHI = preprocessing_parameters.NPY_PATH
PATHIT = PATHI+'inputLST'
PATHIC = PATHI+'inputLSTC'
PATHO = preprocessing_parameters.TFR_PATH


# Calculate standardization as in paper
if preprocessing_parameters.USE_PRECOMPUTED_STDIZATION:
    print("calculating standardization parameters")
    stddevs = np.zeros([3,7,2,361-wsize+1,720-wsize+1])
    means = np.zeros([3,7,2,361-wsize+1,720-wsize+1])
    for i in range(1999, 2014):
        npa = np.zeros([105,3,7,2,361,720])
        for j in range(0,105):
            npa[j,:,:,:,:,:] = np.load(PATHIT+str(i)+str(j)+'.npy')[:,0,:,:,:,:]
        for j in range(361-wsize+1):
            for k in range(720-wsize+1):
                means[:,:,:,j,k] += np.mean(npa[:,:,:,:,j:j+wsize,k:k+wsize],axis=(0,4,5))
                stddevs[:,:,:,j,k] += np.std(npa[:,:,:,:,j:j+wsize,k:k+wsize],axis=(0,4,5))
        del npa
    means /= 15
    stddevs /= 15

    # change dimension back py padding with 'edge' and applying gaussian filter
    means = np.pad(means, [(0,0),(0,0),(0,0),(int(wsize/2),int(wsize/2)),(int(wsize/2),int(wsize/2))],'edge')
    stddevs = np.pad(stddevs, [(0,0),(0,0),(0,0),(int(wsize/2),int(wsize/2)),(int(wsize/2),int(wsize/2))],'edge')
    means = gaussian_filter1d(means, sigma=10, axis=-1,mode='wrap') # wrap around as it it is the longitude
    means = gaussian_filter1d(means, sigma=10, axis=-2,mode='nearest')
    stddevs = gaussian_filter1d(stddevs, sigma=10, axis=-1,mode='wrap') # wrap around as it it is the longitude
    stddevs = gaussian_filter1d(stddevs, sigma=10, axis=-2,mode='nearest')
    np.save('stddevs',stddevs)
    np.save('means',means)
    print("done with mean and stddev calculation")
else:
    means = np.load('means.npy')
    stddevs = np.load('stddevs.npy')

if not os.path.exists(PATHO+'/train/'):
    os.makedirs(PATHO+'/train/')
if not os.path.exists(PATHO+'/val/'):
    os.makedirs(PATHO+'/val/')
if not os.path.exists(PATHO+'/test/'):
    os.makedirs(PATHO+'/test/')

# Transform data from npy to tfr
for iy in range(1999, 2018):
    OUTPUT = PATHO+'/train/DATA'+str(iy)
    OUTPUTV = PATHO+'/val/DATA'+str(iy)
    OUTPUTTE = PATHO + '/test/DATA' + str(iy)
    if iy in [2014,2015]:
        OUT = OUTPUTV
    elif iy in [2016,2017]:
        OUT = OUTPUTTE
    else:
        OUT = OUTPUT
    writer = tf.io.TFRecordWriter(OUT)
    if preprocessing_parameters.CRPS_PREPROCESSING:
        if preprocessing_parameters.CRPS_TEMP:
            cor = np.load(preprocessing_parameters.CRPS_TEMP_PATH + str(iy) + '.npy')
        else:
            cor = np.load(preprocessing_parameters.CRPS_GEOP_PATH + str(iy) + '.npy')
    if iy == 2017 and preprocessing_parameters.CRPS_PREPROCESSING: # Special case, as there is no ground truth for dates in 2018, therefore the last prediction cannot be trained on / used for evaluation
        ra = 104
    else:
        ra = 105
    for jy in range(0,ra):
        INPUT_T = PATHI+'/inputLST'+str(iy)+str(jy)+'.npy'
        INPUT_TC = PATHI+'/inputLSTC'+str(iy)+str(jy)+'.npy'
        train_datax = np.load(INPUT_T)
        stdd = np.std(train_datax, axis=1)
        # standardize trajectory 0
        train_datax = train_datax[:,0,:,:,:,:]
        train_datax -= means
        train_datax /= stddevs
        np.reshape(train_datax, (3,7,2,361,720))
        # concatenate standardized trajectory 0 and non-standardized stddevs (spreads)
        train_datax = np.concatenate((train_datax,stdd),axis=1)
        train_datac = cor[jy,2,:,:]-cor[jy,0,:,:] if preprocessing_parameters.CRPS_PREPROCESSING else np.load(INPUT_TC)
        Npy2TFRecord(train_datax, train_datac, writer)
    writer.close()
    print("done with year "+str(iy))
