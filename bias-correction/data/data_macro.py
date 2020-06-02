import sys
sys.path.append('../')
import global_macros

ROOT_DIRECTORY = global_macros.ROOT_DIRECTORY
DATA_DIRECTORY = ROOT_DIRECTORY + "/data"
GRIB_DATA_DIRECTORY = DATA_DIRECTORY + "/initdata"
NPY_DATA_DIRECTORY = DATA_DIRECTORY + "/npydata"
TF_DATA_DIRECTORY = DATA_DIRECTORY + "/tfdata"

Nheight = 7  # 7 hPa levels
Nparamens = 6
Nzparam = 4
Nlatitude = 41
Nlongitude = 141
Ni = 361
Nj = 720
Nm = 10

height_L = ['150', '200', '250', '400', '500', '700', '850']
type_L = ['t']

DictInvert = lambda d: dict(zip(d.values(), d.keys()))

type2index = {
't' : 0,  # temperature
'u' : 1,  # horizontal wind
'v' : 2,  # vertical wind
'z' : 3,   # geopotential
'r' : 4,  # relative humidity
'cc' : 5 # cloud coverage
}

index2type = DictInvert(type2index)

alevel2index = {
10 : 0, 50 : 1, 100 : 2, 200 : 3,
300 : 4, 400 : 5, 500 : 6, 700: 7, 850 : 8, 925 : 9, 1000 : 10
}

index2alevel = DictInvert(alevel2index)

level2index = {
150: 0, 200: 1, 250: 2, 400: 3,
500: 4, 700: 5, 850: 6
}

index2level = DictInvert(level2index)

zparam2index = {
'2t' : 0,
'tp' : 1,
'100u' : 2,
'100v' : 3
}

index2zparam = DictInvert(zparam2index)

zlevel2index = {
0 : 0
}
index2zlevel = DictInvert(zlevel2index)



global_sfc_type2index = {
'2t' : 0,
# 'tp' : 1,
'skt' : 3,
'sst' : 4,
'tcw' : 5,
'tcwv': 6,
# 'cp' : 6,
'msl' : 7,
'tcc' : 8,
'10u' : 1,
'10v' : 2
}
global_sfc_index2type = DictInvert(global_sfc_type2index)

global_sfc_level2index = {
0 : 0
}
global_sfc_index2level = DictInvert(global_sfc_level2index)


global_pl_type2index = {
't' : 0,
'q' : 1,
'w' : 2,
'd' : 3,
'u' : 4,
'v' : 5,
'z' : 6
}
global_pl_index2type = DictInvert(global_pl_type2index)

global_pl_level2index = {
10 : 0, 50 : 1, 100 : 2, 200 : 3,
300 : 4, 400 : 5, 500 : 6, 700: 7, 850 : 8, 925 : 9, 1000 : 10
}
global_pl_index2level = DictInvert(global_pl_level2index)