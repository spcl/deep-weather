import numpy as np
import glob


def combine_pred():
    years = np.arange(2016, 2018)
    fns = glob.glob("./*.npy")
    data_shape = np.load(fns[0]).shape
    nfiles = len(fns)

    for year in years:
        ofn = str(year)
        data = []
        dates = []
        for fn in fns:
            if str(year) in fn:
                dates.append(fn)
        dates.sort()
        for fn in dates:
            data.append(np.load(fn))
        gather = np.stack(data)
        np.save(ofn, gather)
        print(dates)


def convert_mean_std():
    new_mean_path = "means.npy"
    new_std_path = "stddevs.npy"
    cor_array = [0, 1, 2, 4, 5, 6, 3]
    mean500 = "mean_pl500slice48"
    std500 = "std_pl500slice48"
    mean850 = "mean_pl850slice48"
    std850 = "std_pl850slice48"
    new_mean_vec = np.mean(np.load(new_mean_path), axis=0, keepdims=True)
    new_std_vec = np.mean(np.load(new_std_path), axis=0, keepdims=True)

    mean500_vec = np.zeros([7, 1, 361, 720])
    mean850_vec = np.zeros([7, 1, 361, 720])
    std500_vec = np.zeros([7, 1, 361, 720])
    std850_vec = np.zeros([7, 1, 361, 720])

    for i in range(7):
        mean500_vec[cor_array[i], :, :, :] = new_mean_vec[0, i:i+1, 0:1, :, :]
        std500_vec[cor_array[i], :, :, :] = new_std_vec[0, i:i+1, 0:1, :, :]
        mean850_vec[cor_array[i], :, :, :] = new_mean_vec[0, i:i+1, 1:2, :, :]
        std850_vec[cor_array[i], :, :, :] = new_std_vec[0, i:i+1, 1:2, :, :]

    np.save(mean500, mean500_vec)
    np.save(std500, std500_vec)
    np.save(mean850, mean850_vec)
    np.save(std850, std850_vec)





