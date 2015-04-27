from fuel.transformers import Transformer, Filter, Mapping
import numpy
import theano
import random

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
    id_polyline=stream.sources.index('polyline')
    def first_k(x):
        return (numpy.array(x[id_polyline][:k], dtype=theano.config.floatX).flatten(),)
    stream = Filter(stream, lambda x: len(x[id_polyline])>=k)
    stream = Mapping(stream, first_k, ('first_k',))
    return stream

def add_random_k(k, stream):
    id_polyline=stream.sources.index('polyline')
    def random_k(x):
        loc = random.randrange(len(x[id_polyline])-k+1)
        return (numpy.array(x[id_polyline][loc:loc+k], dtype=theano.config.floatX).flatten(),)
    stream = Filter(stream, lambda x: len(x[id_polyline])>=k)
    stream = Mapping(stream, random_k, ('last_k',))
    return stream

def add_last_k(k, stream):
    id_polyline=stream.sources.index('polyline')
    def last_k(x):
        return (numpy.array(x[id_polyline][-k:], dtype=theano.config.floatX).flatten(),)
    stream = Filter(stream, lambda x: len(x[id_polyline])>=k)
    stream = Mapping(stream, last_k, ('last_k',))
    return stream

def add_destination(stream):
    id_polyline=stream.sources.index('polyline')
    return Mapping(stream, lambda x: (numpy.array(x[id_polyline][-1], dtype=theano.config.floatX),), ('destination',))

def concat_destination_xy(stream):
    id_dx=stream.sources.index('destination_x')
    id_dy=stream.sources.index('destination_y')
    return Mapping(stream, lambda x: (numpy.array([x[id_dx], x[id_dy]], dtype=theano.config.floatX),), ('destination',))
