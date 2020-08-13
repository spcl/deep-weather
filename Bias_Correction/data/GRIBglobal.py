import numpy as np
import data_macro as data_macro
from tqdm import trange
from eccodes import *
import tensorflow as tf
import os

# MACRO
SR = 48
X_SUFFIX = "_x0"
Y_SUFFIX = "_y" + str(SR)
FILE_COMMENT = "plslice"
XONLY_FLAG = True
INCLUDE_Y = False   # Only if XONLY_FLAG is true, set to true if Y is in the data. XONLY_FLAG false doe not matter


def meanparse(array):
    # return np.mean(array, axis=4)  # mean of 10 ensembles
    return np.mean(array[:,:,:,:,:5], axis=4)  # mean of the first 5 ensembles


def GRIBextract(ifn, opathpre, type_L, height_L, type2index, level2index, parse=meanparse):
    '''
    :param ifn: input file name
    :param opathpre: output path prefix
    :param type_L: list of type to extract (string)
    :param height_L: list of heights to extract (string)
    :param type2index: dictionary for type index mapping
    :param level2index: dictionary for level index mapping
    :param parse: callback for ENS processing
    '''
    # XONLY_FLAG and INCLUDE_Y are used to incoporate
    INPUT_FILE = PATHI + ifn
    rf = open(INPUT_FILE)
    ncodes = codes_count_in_file(rf)

    Nheight = len(height_L)
    Nparamens = len(type_L)
    Ni = data_macro.Ni  # latitude
    Nj = data_macro.Nj  # longitude
    Nm = data_macro.Nm  # number of members

    currentDate = -1
    x0avg = None
    ySRavg = None

    for ise in trange(ncodes):
        gidm = codes_grib_new_from_file(rf)
        dataDate = codes_get(gidm, "dataDate")
        sn = codes_get(gidm, "shortName")
        height = codes_get(gidm, "level")
        member = codes_get(gidm, "number") - 1
        sr = int(codes_get(gidm, "stepRange"))

        if dataDate != currentDate:
            if x0avg is not None:
                opre = PATHO + opathpre + "_" + str(currentDate)
                ofnx = opre + X_SUFFIX
                assert(not np.isnan(x0avg).any())
                x0avg = parse(x0avg)
                np.save(ofnx, x0avg)

                if not (XONLY_FLAG and not INCLUDE_Y):
                    ofny = opre + Y_SUFFIX
                    assert(not np.isnan(ySRavg).any())
                    ySRavg = parse(ySRavg)
                    np.save(ofny, ySRavg)

            if XONLY_FLAG:
                x0avg = np.full([Nparamens, Nheight, Ni, Nj, 1], np.nan, dtype=np.float32)
                if INCLUDE_Y:
                    ySRavg = np.full([Nparamens, Nheight, Ni, Nj, 1], np.nan, dtype=np.float32)
            else:
                x0avg = np.full([Nparamens, Nheight, Ni, Nj, Nm], np.nan, dtype=np.float32)
                ySRavg = np.full([Nparamens, Nheight, Ni, Nj, Nm], np.nan, dtype=np.float32)

            currentDate = dataDate

        if sn in type_L and height in height_L:
            typeind = type2index[sn]
            heightind = level2index[height]
            values = codes_get_values(gidm)
            if sr == 0:
                x0avg[typeind, heightind, :, :, member] = np.copy(values.reshape([Ni, Nj]))
            elif sr == SR:
                ySRavg[typeind, heightind, :, :, member] = np.copy(values.reshape([Ni, Nj]))
        codes_release(gidm)

    # save the last date
    if x0avg is not None:
        opre = PATHO + opathpre + "_" + str(currentDate)
        ofnx = opre + X_SUFFIX
        assert (not np.isnan(x0avg).any())
        x0avg = parse(x0avg)
        np.save(ofnx, x0avg)
        if not (XONLY_FLAG and not INCLUDE_Y):
            ofny = opre + Y_SUFFIX
            assert (not np.isnan(ySRavg).any())
            ySRavg = parse(ySRavg)
            np.save(ofny, ySRavg)
    return 0


def GRIBprocess(files, opathpre, type_L, height_L, type2index, level2index, parse=meanparse):
    for ifn in files:
        print("Processing: " + ifn)
        GRIBextract(ifn, opathpre, type_L, height_L, type2index, level2index, parse)


def getfilelist(rf):
    fnl = []
    with open(rf, "r") as rfh:
        # for i, l in enumerate(rfh):
        for l in rfh:
            fnl.append(l.strip())
    return fnl


def GRIBsfc(files, opathpre="sfc"):
    types = data_macro.global_sfc_type2index.keys()
    heights = data_macro.global_sfc_level2index.keys()
    GRIBprocess(files, opathpre=opathpre, type_L=types, height_L=heights,
        type2index=data_macro.global_sfc_type2index, level2index=data_macro.global_sfc_level2index)


def GRIBpl(files, opathpre="pl"):
    types = data_macro.global_pl_type2index.keys()
    heights = data_macro.global_pl_level2index.keys()
    GRIBprocess(files, opathpre=opathpre, type_L=types, height_L=heights,
        type2index=data_macro.global_pl_type2index, level2index=data_macro.global_pl_level2index)


def GRIBplslice(files, opathpre="plslice", level=850):
    types = ['t', 'q', 'w', 'd', 'u', 'v', 'z']
    heights = [level]
    GRIBprocess(files, opathpre=opathpre, type_L=types, height_L=heights,
                type2index=data_macro.global_pl_type2index, level2index={level: 0})


def getMeanStd(ipath, file_comment, normdim=(2, 3)):
    import glob
    import os

    mean_file_path = ipath + "/mean_" + file_comment + ".npy"
    std_file_path = ipath + "/std_" + file_comment + ".npy"

    if os.path.isfile(mean_file_path):
        os.remove(mean_file_path)
    if os.path.isfile(std_file_path):
        os.remove(std_file_path)

    fns = glob.glob(ipath + "/*.npy")
    data_shape = np.load(fns[0]).shape
    meansum = np.zeros(data_shape)
    sqrsum = np.zeros(data_shape)
    nfiles = len(fns)
    for i in fns:
        data = np.load(i)
        meansum += data
        sqrsum += np.square(data)
    meanvec = np.mean(meansum / nfiles, axis=normdim, keepdims=True)
    stdvec = np.sqrt(np.mean(sqrsum / nfiles - np.square(meanvec), axis=normdim, keepdims=True))
    # sqrsum = np.zeros(data_shape)
    # for i in fns:
    #     data = np.load(i)
    #     sqrsum += np.square(data - meanvec)
    # stdvec = np.sqrt( np.mean(sqrsum / nfiles, axis=normdim, keepdims=True) )
    np.save(mean_file_path, meanvec)
    np.save(std_file_path, stdvec)


def get_split_string(fstr):
    flist = fstr.split("_")
    suffix = flist[-1]
    datastr = flist[-2]
    prefix = "_".join(flist[:-2])
    return prefix, datastr, suffix



def getNextDateFile(filename, incr = 1):
    from datetime import date
    assert(incr < 30), "Not supporting large date increment for now"
    fsl = filename.split(".")
    extension = fsl[-1]
    fstr = ".".join(fsl[:-1])

    prefix, datestr, suffix = get_split_string(fstr)
    assert(len(datestr) == 8)
    yy = int(datestr[:4])
    mm = int(datestr[4:6])
    dd = int(datestr[6:8])
    c_date = date(yy, mm, dd)
    increment = date(1990, 1, 1+incr) - date(1990, 1, 1)
    newdate = c_date + increment
    yystr = str(newdate.year)
    mmstr = str(newdate.month)
    ddstr = str(newdate.day)
    if len(mmstr) == 1:
        mmstr = "0" + mmstr
    if len(ddstr) == 1:
        ddstr = "0" + ddstr
    newfn = prefix + "_" + yystr + mmstr + ddstr + "_" + suffix + "." + extension
    return newfn


def getSelectedSuffix(filename, suffix):
    fsl = filename.split(".")
    # assert(len(fsl) == 2), "File name containing not exactly one period (.)"
    extension = fsl[-1]
    fstr = ".".join(fsl[:-1])
    prefix, datestr, _ = get_split_string(fstr)
    newfn = prefix + "_" + datestr + "_" + suffix + "." + extension
    return newfn


# HACK
# The suffix and incr here are subjective to change
def getSample(filename):
    suffix = "y" + str(SR)
    y48 = getSelectedSuffix(filename, suffix)
    x48 = getNextDateFile(filename, incr=int(SR/24))
    return [filename, y48, x48]


def getxyfilenames(ipath):
    import glob
    fns = glob.glob(ipath + "/*_x0.npy")
    xyfns = []
    for fn in fns:
        samplefiles = getSample(fn)
        if os.path.isfile(samplefiles[1]) and os.path.isfile(samplefiles[2]):
            xyfns.append(samplefiles)
    return xyfns


def makeTFRecordyearly(ipath, opath, xyfns, year, file_comment):
    writer = tf.python_io.TFRecordWriter(opath + "tf_" + str(year) + "_" + file_comment)
    size = len(xyfns)
    mean_file_path = ipath + "/mean_" + file_comment + ".npy"
    std_file_path = ipath + "/std_" + file_comment + ".npy"
    mean = np.load(mean_file_path)
    std = np.load(std_file_path)
    assert (os.path.isfile(mean_file_path)), "Mean file does not exist!"
    assert (os.path.isfile(std_file_path)), "Std file does not exist!"

    fct = 0
    for h in trange(0, size):
        if "_" + str(year) in xyfns[h][0]:  # check if year is in the filename
            x0 = np.load(xyfns[h][0])
            y48 = np.load(xyfns[h][1])
            x48 = np.load(xyfns[h][2])
            x0 = (x0 - mean) / std
            y48 = (y48 - mean) / std
            x48 = (x48 - mean) / std
            # the first 7 channels are the current reanalysis (x0), the latter 7 are forecasts (y48)
            X = np.concatenate([x0, y48], axis=0)
            # the output is the reanalysis of 48 hours later
            Y = x48
            features = {
                'Date': tf.train.Feature(int64_list=tf.train.Int64List(value=[int(xyfns[h][0].split("_")[-2])])),
                'X': tf.train.Feature(float_list=tf.train.FloatList(value=X.flatten())),
                'Y': tf.train.Feature(float_list=tf.train.FloatList(value=Y.flatten()))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            fct += 1
    print(str(fct) + " data points extracted for year " + str(year))
    writer.close()


def makeTFRecord(ipath, opath, file_comment):
    xyfns = getxyfilenames(ipath)
    print("The number of data point is: ", len(xyfns))
    years = np.arange(1998, 2018)  # change year range
    for year in years:
        makeTFRecordyearly(ipath, opath, xyfns, year, file_comment)


if __name__ == "__main__":
    # Define paths here:
    ensemble_GRIB_path = "./ENS10_data/"  # folder containing ENS10 GRIB files (keep the slash at the end)
    analysis_GRIB_path = "./ERA5_data/"   # folder containing ERA5 GRIB files (keep the slash at the end)
    npy_output_path = "./npydata/"        # folder to contain output the intermediate numpy files (need the folder to exist, keep the slash at the end)
    tfrecord_output_path = "./tfdata/"    # folder to contain tfrecord files (need the folder to exist, keep the slash at the end)
    slice_level = 850                     # pressure level of the slice (850 or 500)

    # main script:
    # ENS10 dataset GRIB to npy
    file_comment = "pl" + str(slice_level) + "slice48"
    files = getfilelist("plfl")
    PATHI = ensemble_GRIB_path
    PATHO = npy_output_path
    PATHTF = tfrecord_output_path
    XONLY_FLAG = False
    GRIBplslice(files, level=slice_level)
    # Renanalysis GRIB to npy
    files = getfilelist("planalysis")
    PATHI = analysis_GRIB_path
    PATHO = npy_output_path
    PATHTF = tfrecord_output_path
    XONLY_FLAG = True
    GRIBplslice(files, level=slice_level)
    # get mean and std
    getMeanStd(PATHO,file_comment)
    # npy to TFRecord
    out = makeTFRecord(ipath=PATHO, opath=PATHTF, file_comment=file_comment)
