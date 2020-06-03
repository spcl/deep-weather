#!/usr/bin/env python2

# Needs to be run in python2 for eccodes to work

import numpy as np
import multiprocessing as mp
import preprocessing_parameters
from eccodes import *
file_indexes = ['0101', '0104', '0108', '0111', '0115', '0118', '0122', '0125', '0129', '0201', '0205', '0208', '0212', '0215', '0219', '0222', '0226', '0301', '0305', '0308', '0312', '0315', '0319', '0322', '0326', '0329', '0402', '0405', '0409', '0412', '0416', '0419', '0423', '0426', '0430', '0503', '0507', '0510', '0514', '0517', '0521', '0524', '0528', '0531', '0604', '0607', '0611', '0614', '0618', '0621', '0625', '0628', '0702', '0705', '0709', '0712', '0716', '0719', '0723', '0726', '0730', '0802', '0806', '0809', '0813', '0816', '0820', '0823', '0827', '0830', '0903', '0906', '0910', '0913', '0917', '0920', '0924', '0927', '1001', '1004', '1008', '1011', '1015', '1018', '1022', '1025', '1029', '1101', '1105', '1108', '1112', '1115', '1119', '1122', '1126', '1129', '1203', '1206', '1210', '1213', '1217', '1220', '1224', '1227', '1231']
prefix = 'output.pl.2018'
INPUT_PATH = preprocessing_parameters.GRIB_PATH
OUTPUT_PATH = preprocessing_parameters.NPY_PATH

PARAMCODE = "130.128" # Parameters predicted
params = ['130.128','133.128','135.128','131.128','132.128','129.128','155.128'] # Parameters used
used_pert = [1,2,3,4,5] # Perturbations used for the prediction
total_pert = [1,2,3,4,5,6,7,8,9,10] # Perturbations that the prediction will be compared against
used_plvl = [500,850] # Pressure levels that are used in hPa
## Code for the GRIB parameters if available
# 248.128 = fraction cloud cover
# 157.128 = relative humidity
# 131.128 = U component of wind
# 132.128 = V component of wind
# 129.128 = Geopotential
# 130.128 = Temperature
# 155.128 = Divergence
# 133.128 = Specific humidity
# 135.128 = Vertical velocity

# Extract information from GRIB files using ECCODES
def calc(i):
    OUTPUT = OUTPUT_PATH + 'inputLST' + str(i)
    OUTPUTC = OUTPUT_PATH + 'inputLSTC' + str(i)
    Nhours = 105
    assert Nhours == len(file_indexes)
    Nparam = len(params)
    Nheight = 2
    Nlatitude = 361
    Nlongitude = 720
    npx = np.empty([3, len(used_pert), Nparam, Nheight, Nlatitude, Nlongitude],dtype='float32')
    npc = np.empty([len(total_pert),Nparam,Nheight, Nlatitude, Nlongitude],dtype='float32')
    for hi,ind in enumerate(file_indexes):
        FILE = INPUT_PATH+prefix+ind
        with open(FILE, "r") as fd:

            print(str(i)+' Year and index '+str(hi+1)+'/'+str(Nhours))
            for ise in xrange(0, 46179): # hardcoded the number of GRIB messages per file, to avoid an initial iteration through the whole file
                gidt = codes_grib_new_from_file(fd) # Get the next GRIB message
                y = codes_get(gidt, "year")
                lvl = codes_get(gidt, "ls.level")
                if i == y and lvl in used_plvl: #only look at it if we're in the correct year, only 500 and 850hpa for now
                    yi = used_plvl.index(lvl);
                    P1t = codes_get(gidt, "P1")
                    pnr = codes_get(gidt, "perturbationNumber")
                    Paramt = codes_get(gidt, "param")
                    values = codes_get_values(gidt)

                    if P1t == 0:
                        if pnr in used_pert:
                            pert = used_pert.index(pnr);
                            paramit = params.index(Paramt)
                            npx[0, pert, paramit, yi, :, :] = np.reshape(values,(Nlatitude,Nlongitude))

                    elif P1t == 24:
                        if pnr in used_pert:
                            pert = used_pert.index(pnr);
                            paramit = params.index(Paramt)
                            npx[1, pert, paramit, yi, :, :] = np.reshape(values,(Nlatitude,Nlongitude))

                    elif P1t == 48:
                        pert = total_pert.index(pnr)
                        paramit = params.index(Paramt)
                        npc[pert,paramit, yi, :, :] = np.reshape(values,(Nlatitude,Nlongitude))
                        if pnr in used_pert:
                            pert = used_pert.index(pnr);
                            paramit = params.index(Paramt)
                            npx[2, pert, paramit, yi, :, :] = np.reshape(values,(Nlatitude,Nlongitude))

                codes_release(gidt)
            cstddev = np.std(npc, axis=0, keepdims=True)
            cmean = np.mean(npc, axis=0, keepdims=True)
            train = npx
            correct = np.concatenate((cstddev,cmean),axis=0)
            np.save(OUTPUT+str(hi),train)
            np.save(OUTPUTC+str(hi), correct)
    print("done with " + str(i))


# Preprocess GRIB files in parallel, set the amount of workers here
pool = mp.Pool(5) # Uses a lot of RAM per worker
pool.map(calc, list(range(1999,2018)))
