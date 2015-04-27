from theano import tensor
import numpy


def hdist(a, b):
    rearth = numpy.float32(6371)
    deg2rad = numpy.float32(3.14159265358979 / 180)

    lat1 = a[:, 1] * deg2rad
    lon1 = a[:, 0] * deg2rad
    lat2 = b[:, 1] * deg2rad
    lon2 = b[:, 0] * deg2rad

    dlat = abs(lat1-lat2)
    dlon = abs(lon1-lon2)

    al = tensor.sin(dlat/2)**2  + tensor.cos(lat1) * tensor.cos(lat2) * (tensor.sin(dlon/2)**2)
    d = tensor.arctan2(tensor.sqrt(al), tensor.sqrt(numpy.float32(1)-al))

    hd = 2 * rearth * d

    return tensor.switch(tensor.eq(hd, float('nan')), (a-b).norm(2, axis=1), hd)



