    random_crop,
    horizontal_flip,
    vertical_flip,
    transpose,

import random

def reduce_sample_y(data_y, args):
    ind = 1 if args.aggr_type=="Mean" else 0
    data_y = data_y[ind,None,args.parameters.index(args.param_type),:,:,:]
    return data_y

def random_crop(data_x, data_y, args):
    max_lat = data_y.shape[-2]-args.crop_lat
    max_lon = data_y.shape[-1]-args.crop_lon
    lat = random.randint(0,max_lat)
    lon = random.randint(0,max_lon)
    data_x = data_x[:,:,lat:lat+args.crop_lat,lon:lon+args.crop_lon]
    data_y = data_y[:,:,lat:lat+args.crop_lat,lon:lon+args.crop_lon]

    return data_x, data_y

def horizontal_flip(data_x, data_y, args):
    if random.random() < 0.5:
        data_x = np.flip(data_x,-1)
        data_y = np.flip(data_y,-1)
    return data_x, data_y

def vertical_flip(data_x, data_y, args):
    if random.random() < 0.5:
        data_x = np.flip(data_x,-2)
        data_y = np.flip(data_y,-2)
    return data_x, data_y

def transpose(data_x, data_y, args):
    if random.random() < 0.5:
        data_x = data_x.transpose(0,1,3,2)
        data_y = data_y.transpose(0,1,3,2)
    return data_x, data_y
    #TODO check if data needs to be contiguous, if yes, use .copy()