import datetime

import numpy
import theano

import fuel

from fuel.schemes import ConstantScheme
from fuel.transformers import Batch, Mapping, SortMapping, Transformer, Unpack, FilterSources

import data

fuel.config.default_seed = 123

def at_least_k(k, v, pad_at_begin, is_longitude):
    if len(v) == 0:
        v = numpy.array([data.train_gps_mean[1 if is_longitude else 0]], dtype=theano.config.floatX)
    if len(v) < k:
        if pad_at_begin:
            v = numpy.concatenate((numpy.full((k - len(v),), v[0]), v))
        else:
            v = numpy.concatenate((v, numpy.full((k - len(v),), v[-1])))
    return v

Select = FilterSources

class TaxiExcludeTrips(Transformer):
    produces_examples = True

    def __init__(self, stream, exclude_list):
        super(TaxiExcludeTrips, self).__init__(stream)
        self.id_trip_id = stream.sources.index('trip_id')
        self.exclude = {v: True for v in exclude_list}
    def get_data(self, request=None):
        if request is not None: raise ValueError
        while True:
            data = next(self.child_epoch_iterator)
            if not data[self.id_trip_id] in self.exclude: break
        return data

class TaxiExcludeEmptyTrips(Transformer):
    produces_examples = True

    def __init__(self, stream):
        super(TaxiExcludeEmptyTrips, self).__init__(stream)
        self.latitude = stream.sources.index('latitude')
    def get_data(self, request=None):
        if request is not None: raise ValueError
        while True:
            data = next(self.child_epoch_iterator)
            if len(data[self.latitude])>0: break
        return data
        
class TaxiGenerateSplits(Transformer):
    produces_examples = True

    def __init__(self, data_stream, max_splits=-1):
        super(TaxiGenerateSplits, self).__init__(data_stream)

        self.sources = data_stream.sources 
        if not data.tvt:
            self.sources += ('destination_latitude', 'destination_longitude', 'travel_time')
        self.max_splits = max_splits
        self.data = None
        self.splits = []
        self.isplit = 0
        self.id_latitude = data_stream.sources.index('latitude')
        self.id_longitude = data_stream.sources.index('longitude')

        self.rng = numpy.random.RandomState(fuel.config.default_seed)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        while self.isplit >= len(self.splits):
            self.data = next(self.child_epoch_iterator)
            self.splits = range(len(self.data[self.id_longitude]))
            self.rng.shuffle(self.splits)
            if self.max_splits != -1 and len(self.splits) > self.max_splits:
                self.splits = self.splits[:self.max_splits]
            self.isplit = 0
        
        i = self.isplit
        self.isplit += 1
        n = self.splits[i]+1

        r = list(self.data)

        r[self.id_latitude] = numpy.array(r[self.id_latitude][:n], dtype=theano.config.floatX)
        r[self.id_longitude] = numpy.array(r[self.id_longitude][:n], dtype=theano.config.floatX)

        r = tuple(r)

        if data.tvt:
            return r
        else:
            dlat = numpy.float32(self.data[self.id_latitude][-1])
            dlon = numpy.float32(self.data[self.id_longitude][-1])
            ttime = numpy.int32(15 * (len(self.data[self.id_longitude]) - 1))
            return r + (dlat, dlon, ttime)

class _taxi_add_first_last_len_helper(object):
    def __init__(self, k, id_latitude, id_longitude):
        self.k = k
        self.id_latitude = id_latitude
        self.id_longitude = id_longitude
    def __call__(self, data):
        first_k = (numpy.array(at_least_k(self.k, data[self.id_latitude], False, False)[:self.k],
                               dtype=theano.config.floatX),
                   numpy.array(at_least_k(self.k, data[self.id_longitude], False, True)[:self.k],
                               dtype=theano.config.floatX))
        last_k = (numpy.array(at_least_k(self.k, data[self.id_latitude], True, False)[-self.k:],
                            dtype=theano.config.floatX),
                  numpy.array(at_least_k(self.k, data[self.id_longitude], True, True)[-self.k:],
                              dtype=theano.config.floatX))
        input_time = (numpy.int32(15 * (len(data[self.id_latitude]) - 1)),)
        return first_k + last_k + input_time

def taxi_add_first_last_len(stream, k):
    fun = _taxi_add_first_last_len_helper(k, stream.sources.index('latitude'), stream.sources.index('longitude'))
    return Mapping(stream, fun, add_sources=('first_k_latitude', 'first_k_longitude', 'last_k_latitude', 'last_k_longitude', 'input_time'))


class _taxi_add_datetime_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        ts = data[self.key]
        date = datetime.datetime.utcfromtimestamp(ts)
        yearweek = date.isocalendar()[1] - 1
        info = (numpy.int8(51 if yearweek == 52 else yearweek),
                numpy.int8(date.weekday()),
                numpy.int8(date.hour * 4 + date.minute / 15))
        return info

def taxi_add_datetime(stream):
    fun = _taxi_add_datetime_helper(stream.sources.index('timestamp'))
    return Mapping(stream, fun, add_sources=('week_of_year', 'day_of_week', 'qhour_of_day'))


class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return data[self.key].shape[0]

def balanced_batch(stream, key, batch_size, batch_sort_size):
    stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size * batch_sort_size))
    comparison = _balanced_batch_helper(stream.sources.index(key))
    stream = Mapping(stream, SortMapping(comparison))
    stream = Unpack(stream)
    return Batch(stream, iteration_scheme=ConstantScheme(batch_size))


class _taxi_remove_test_only_clients_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, x):
        x = list(x)
        if x[self.key] >= data.origin_call_train_size:
            x[self.key] = numpy.int32(0)
        return tuple(x)

def taxi_remove_test_only_clients(stream):
    fun = _taxi_remove_test_only_clients_helper(stream.sources.index('origin_call'))
    return Mapping(stream, fun)


class _add_destination_helper(object):
    def __init__(self, latitude, longitude):
        self.latitude = latitude
        self.longitude = longitude
    def __call__(self, data):
        return (data[self.latitude][-1], data[self.longitude][-1])

def add_destination(stream):
    fun = _add_destination_helper(stream.sources.index('latitude'), stream.sources.index('longitude'))
    return Mapping(stream, fun, add_sources=('destination_latitude', 'destination_longitude'))

class _window_helper(object):
    def __init__(self, latitude, longitude, window_len):
        self.latitude = latitude
        self.longitude = longitude
        self.window_len = window_len
    def makewindow(self, x):
        assert len(x.shape) == 1

        if x.shape[0] < self.window_len:
            x = numpy.concatenate(
                [numpy.full((self.window_len - x.shape[0],), x[0]), x])
            
        y = [x[i: i+x.shape[0]-self.window_len+1][:, None]
             for i in range(self.window_len)]

        return numpy.concatenate(y, axis=1)

    def __call__(self, data):
        data = list(data)
        data[self.latitude] = self.makewindow(data[self.latitude])
        data[self.longitude] = self.makewindow(data[self.longitude])
        return tuple(data)


def window(stream, window_len):
    fun = _window_helper(stream.sources.index('latitude'),
                         stream.sources.index('longitude'),
                         window_len)
    return Mapping(stream, fun)

