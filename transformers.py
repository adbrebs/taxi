from fuel.transformers import Transformer, Filter, Mapping
import numpy
import theano

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

def add_extremities(stream, k):
    id_polyline=stream.sources.index('polyline')
    def extremities(x):
        return (numpy.array(x[id_polyline][:k], dtype=theano.config.floatX).flatten(),
                numpy.array(x[id_polyline][-k:], dtype=theano.config.floatX).flatten())
    stream = Filter(stream, lambda x: len(x[id_polyline])>=k)
    stream = Mapping(stream, extremities, ('first_k', 'last_k'))
    return stream

def add_destination(stream):
    id_polyline=stream.sources.index('polyline')
    return Mapping(stream, lambda x: (numpy.array(x[id_polyline][-1], dtype=theano.config.floatX),), ('destination',))
