from fuel.transformers import Transformer, Filter, Mapping
import numpy
import theano
import random

def at_least_k(k, pl, pad_at_begin):
    if len(pl) == 0:
        pl = [[ -8.61612, 41.1573]]
    if len(pl) < k:
        if pad_at_begin:
            pl = [pl[0]] * (k - len(pl)) + pl
        else:
            pl = pl + [pl[-1]] * (k - len(pl))
    return pl


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
        pl = at_least_k(k, x[id_polyline], False)
        return (numpy.array(pl[:k], dtype=theano.config.floatX).flatten(),)
    stream = Mapping(stream, first_k, ('first_k',))
    return stream

def add_random_k(k, stream):
    id_polyline=stream.sources.index('polyline')
    def random_k(x):
        pl = at_least_k(k, x[id_polyline], True)
        loc = random.randrange(len(pl)-k+1)
        return (numpy.array(pl[loc:loc+k], dtype=theano.config.floatX).flatten(),)
    stream = Mapping(stream, random_k, ('last_k',))
    return stream

def add_last_k(k, stream):
    id_polyline=stream.sources.index('polyline')
    def last_k(x):
        pl = at_least_k(k, x[id_polyline], True)
        return (numpy.array(pl[-k:], dtype=theano.config.floatX).flatten(),)
    stream = Mapping(stream, last_k, ('last_k',))
    return stream

def add_destination(stream):
    id_polyline=stream.sources.index('polyline')
    return Mapping(stream,
        lambda x:
            (numpy.array(at_least_k(1, x[id_polyline], True)[-1], dtype=theano.config.floatX),),
        ('destination',))

def concat_destination_xy(stream):
    id_dx=stream.sources.index('destination_x')
    id_dy=stream.sources.index('destination_y')
    return Mapping(stream, lambda x: (numpy.array([x[id_dx], x[id_dy]], dtype=theano.config.floatX),), ('destination',))
