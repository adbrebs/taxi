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
        
class TaxiGenerateSplits(Transformer):
    def __init__(self, data_stream, max_splits=-1):
        super(TaxiGenerateSplits, self).__init__(data_stream)
        self.sources = data_stream.sources + ('destination_latitude', 'destination_longitude')
        self.max_splits = max_splits
        self.data = None
        self.splits = []
        self.isplit = 0
        self.id_latitude = data_stream.sources.index('latitude')
        self.id_longitude = data_stream.sources.index('longitude')

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        while self.isplit >= len(self.splits):
            self.data = next(self.child_epoch_iterator)
            self.splits = range(len(self.data[self.id_polyline]))
            random.shuffle_array(self.splits)
            if self.max_splits != -1 and len(self.splits) > self.max_splits:
                self.splits = self.splits[:self.max_splits]
            self.isplit = 0
        
        i = self.isplit
        self.isplit += 1
        n = self.splits[i]+1

        r = list(self.data)

        r[self.id_latitude] = r[self.id_latitude][:n]
        r[self.id_longitude] = r[self.id_longitude][:n]

        dlat = self.data[self.id_latitude][-1]
        dlon = self.data[self.id_longitude][-1]

        return tuple(r + [dlat, dlon])


class first_k(object):
    def __init__(self, k, id_latitude, id_longitude):
        self.k = k
        self.id_latitude = id_latitude
        self.id_longitude = id_longitude
    def __call__(self, data): 
        return (numpy.array(at_least_k(self.k, data[self.id_latitude], False, False)[:self.k],
                            dtype=theano.config.floatX),
                numpy.array(at_least_k(self.k, data[self.id_longitude], False, True)[:self.k],
                            dtype=theano.config.floatX))
def add_first_k(k, stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    return Mapping(stream, first_k(k, id_latitude, id_longitude), ('first_k_latitude', 'first_k_longitude'))

class random_k(object):
    def __init__(self, k, id_latitude, id_longitude):
        self.k = k
        self.id_latitude = id_latitude
        self.id_longitude = id_longitude
    def __call__(self, x):
        lat = at_least_k(self.k, x[self.id_latitude], True, False)
        lon = at_least_k(self.k, x[self.id_longitude], True, True)
        loc = random.randrange(len(lat)-self.k+1)
        return (numpy.array(lat[loc:loc+self.k], dtype=theano.config.floatX),
                numpy.array(lon[loc:loc+self.k], dtype=theano.config.floatX))
def add_random_k(k, stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    return Mapping(stream, random_k(k, id_latitude, id_longitude), ('last_k_latitude', 'last_k_longitude'))

class last_k(object):
    def __init__(self, k, id_latitude, id_longitude):
        self.k = k
        self.id_latitude = id_latitude
        self.id_longitude = id_longitude
    def __call__(self, data):
        return (numpy.array(at_least_k(self.k, data[self.id_latitude], True, False)[-self.k:],
                            dtype=theano.config.floatX),
                numpy.array(at_least_k(self.k, data[self.id_longitude], True, True)[-self.k:],
                            dtype=theano.config.floatX))
def add_last_k(k, stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    return Mapping(stream, last_k(k, id_latitude, id_longitude), ('last_k_latitude', 'last_k_longitude'))

class destination(object):
    def __init__(self, id_latitude, id_longitude):
        self.id_latitude = id_latitude
        self.id_longitude = id_longitude
    def __call__(self, data):
        return (numpy.array(at_least_k(1, data[self.id_latitude], True, False)[-1],
                            dtype=theano.config.floatX),
                numpy.array(at_least_k(1, data[self.id_longitude], True, True)[-1],
                            dtype=theano.config.floatX))
def add_destination(stream):
    id_latitude = stream.sources.index('latitude')
    id_longitude = stream.sources.index('longitude')
    return Mapping(stream, destination(id_latitude, id_longitude), ('destination_latitude', 'destination_longitude'))


class trip_filter(object):
    def __init__(self, id_trip_id, exclude):
        self.id_trip_id = id_trip_id
        self.exclude = exclude
    def __call__(self, data):
        if data[self.id_trip_id] in self.exclude:
            return False
        else:
            return True
def filter_out_trips(exclude_trips, stream):
    id_trip_id = stream.sources.index('trip_id')
    return Filter(stream, trip_filter(id_trip_id, exclude_trips))

