from theano import tensor
import theano
import numpy

def const(v):
    if theano.config.floatX == 'float32':
        return numpy.float32(v)
    else:
        return numpy.float64(v)

rearth = const(6371)
deg2rad = const(3.141592653589793 / 180)

def hdist(a, b):
    lat1 = a[:, 0] * deg2rad
    lon1 = a[:, 1] * deg2rad
    lat2 = b[:, 0] * deg2rad
    lon2 = b[:, 1] * deg2rad

    dlat = abs(lat1-lat2)
    dlon = abs(lon1-lon2)

    al = tensor.sin(dlat/2)**2  + tensor.cos(lat1) * tensor.cos(lat2) * (tensor.sin(dlon/2)**2)
    d = tensor.arctan2(tensor.sqrt(al), tensor.sqrt(const(1)-al))

    hd = const(2) * rearth * d

    return tensor.switch(tensor.eq(hd, float('nan')), (a-b).norm(2, axis=1), hd)

def erdist(a, b):
    lat1 = a[:, 0] * deg2rad
    lon1 = a[:, 1] * deg2rad
    lat2 = b[:, 0] * deg2rad
    lon2 = b[:, 1] * deg2rad
    x = (lon2-lon1) * tensor.cos((lat1+lat2)/2)
    y = (lat2-lat1)
    return tensor.sqrt(tensor.sqr(x) + tensor.sqr(y)) * rearth

def rmsle(a, b):
    return tensor.sqrt( ( (tensor.log(a+1)-tensor.log(b+1)) ** 2 ).mean() )
