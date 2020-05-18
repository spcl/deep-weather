#from GRIB2NPY import GRIB2NPY
from data.NPY2TF import *
from data.data_macro import *


def main():
    Nyrs = 14
    yrs = np.arange(Nyrs) + 2000
    loadlist = ['X0','X6','Y6']
    #data = loadNPY(yrs = yrs, loadlist=loadlist)
    #data = dimSelect(data, latitudes = np.arange(40), longitudes = np.arange(40))
    data = loadSelect(yrs = yrs, loadlist=loadlist, types=[0]) # latitudes = np.arange(30), longitudes = np.arange(30))
    mean, std = dimNormalize(data, normdim=(0,2,3,4))
    quickNPY2TF(data,yrs,[0,2],[1],comment='_temp') #


def load_main(loadlist, yrs, npformat=False, crop = True, comment='', types = [0], heights = [0], normdim=(0,2,3,4), freq=None):
    if crop:
        data = loadSelect(yrs = yrs, loadlist=loadlist, types = types, heights = heights,
            latitudes = np.arange(40), longitudes = np.arange(136) )
    else:
        data = loadSelect(yrs = yrs, loadlist=loadlist, types = types, heights = heights )
    mean, std = dimNormalize(data, normdim=normdim)
    for i in range(len(data)):
        data[i].append( np.roll(data[i][0], shift=-1, axis=0) )
    for i in range(len(data)):
        for j in range(len(data[0])):
            data[i][j] = data[i][j][:-1]
    print("The example data shape is: ")
    print(data[0][2].shape)
    if npformat:
        quickNPY2NPY(data, yrs, [0,1], [2], comment=comment, freq=freq)
    else:
        quickNPY2TF(data, yrs, [0,1], [2], comment=comment, freq=freq)
    quickMeanStd(mean,std,comment=comment)


def main_sfc(yrs, npformat=False, crop = True, comment='sfc', types = [0], heights = [0], normdim=(0,2,3,4), freq=None):
    loadlist = ['sfc_X0','sfc_X24'] #,'sfc_X48']
    load_main(loadlist, yrs, npformat=npformat, crop=crop, comment=comment,
              types=types, heights=heights, normdim=normdim, freq=freq)


def main_pl(yrs, npformat=False, crop = True, comment='pl', types = np.arange(len(type2index)),
            heights = np.arange(len(alevel2index)), normdim=(0,2,3,4), freq=None):

    loadlist = ['pl_X0','pl_X24'] #,'pl_X48']
    load_main(loadlist, yrs, npformat=npformat, crop=crop, comment=comment,
              types=types, heights=heights, normdim=normdim, freq=freq)


def main_plsfc(yrs, npformat=False, crop = True, comment='plsfc', types = [0],
               heights = np.arange(len(alevel2index)+1), normdim=(0,2,3,4), freq=None):

    loadlist = ['plsfctmp_X0', 'plsfctmp_X24']
    load_main(loadlist, yrs, npformat=npformat, crop=crop, comment=comment,
              types=types, heights=heights, normdim=normdim, freq=freq)




if __name__ == "__main__":
    yrs = [2014, 2015, 2016, 2017, 2018]
    # main_sfc(yrs, npformat=False, comment='sfc', types = [0], heights = [0])
    # main_sfc(yrs, npformat=False, comment='sfc3', types = [0,1,2], heights = [0])
    # main_pl(yrs, npformat=False, comment='pl_24', types = np.arange(4), heights=np.arange(len(alevel2index)))
    # main_plsfc(yrs, npformat=False, comment='plsfc', types = [0], heights = np.arange(len(alevel2index)+1), normdim=(0,3,4))
    main_pl(yrs, npformat=False, comment='pl_pos_24_small', types=np.arange(4), heights=np.arange(len(alevel2index)), freq=[1,3,6,12])