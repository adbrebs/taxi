from fuel.transformers import Transformer, Filter, Mapping
import numpy
import theano
import random
import data

def at_least_k(k, v, pad_at_begin, is_longitude):
    if len(v) == 0:
        v = numpy.array([data.porto_center[1 if is_longitude else 0]], dtype=theano.config.floatX)
    if len(v) < k:
        if pad_at_begin:
            v = numpy.concatenate((numpy.full((k - len(v),), v[0]), v))
        else:
            v = numpy.concatenate((v, numpy.full((k - len(v),), v[-1])))
    return v


class Select(Transformer):
    def __init__(self, data_stream, sources):
        super(Select, self).__init__(data_stream)
        self.ids = [data_stream.sources.index(source) for source in sources]
        self.sources=sources

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data=next(self.child_epoch_iterator)
        return [data[id] for id in self.ids]

def add_first_k(k, stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    return Mapping(stream,
        lambda data:
            (numpy.array(at_least_k(k, data[id_latitude], False, False)[:k], dtype=theano.config.floatX),
             numpy.array(at_least_k(k, data[id_longitude], False, True)[:k], dtype=theano.config.floatX)),
        ('first_k_latitude', 'first_k_longitude'))

def add_random_k(k, stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    def random_k(x):
        lat = at_least_k(k, x[id_latitude], True, False)
        lon = at_least_k(k, x[id_longitude], True, True)
        loc = random.randrange(len(lat)-k+1)
        return (numpy.array(lat[loc:loc+k], dtype=theano.config.floatX),
                numpy.array(lon[loc:loc+k], dtype=theano.config.floatX))
    return Mapping(stream, random_k, ('last_k_latitude', 'last_k_longitude'))

def add_last_k(k, stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    return Mapping(stream,
        lambda data:
            (numpy.array(at_least_k(k, data[id_latitude], True, False)[-k:], dtype=theano.config.floatX),
             numpy.array(at_least_k(k, data[id_longitude], True, True)[-k:], dtype=theano.config.floatX)),
        ('last_k_latitude', 'last_k_longitude'))

def add_destination(stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    return Mapping(stream,
        lambda data:
            (numpy.array(at_least_k(1, data[id_latitude], True, False)[-1], dtype=theano.config.floatX),
             numpy.array(at_least_k(1, data[id_longitude], True, True)[-1], dtype=theano.config.floatX)),
        ('destination_latitude', 'destination_longitude'))
