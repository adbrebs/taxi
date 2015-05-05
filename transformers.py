from fuel.transformers import Transformer, Filter, Mapping
import numpy
import theano
import random
import data

import datetime

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
            self.splits = range(len(self.data[self.id_longitude]))
            random.shuffle(self.splits)
            if self.max_splits != -1 and len(self.splits) > self.max_splits:
                self.splits = self.splits[:self.max_splits]
            self.isplit = 0
        
        i = self.isplit
        self.isplit += 1
        n = self.splits[i]+1

        r = list(self.data)

        r[self.id_latitude] = numpy.array(r[self.id_latitude][:n], dtype=theano.config.floatX)
        r[self.id_longitude] = numpy.array(r[self.id_longitude][:n], dtype=theano.config.floatX)

        dlat = numpy.float32(self.data[self.id_latitude][-1])
        dlon = numpy.float32(self.data[self.id_longitude][-1])

        return tuple(r + [dlat, dlon])

class TaxiAddFirstK(Transformer):
    def __init__(self, k, stream):
        super(TaxiAddFirstK, self).__init__(stream)
        self.sources = stream.sources + ('first_k_latitude', 'first_k_longitude')
        self.id_latitude = stream.sources.index('latitude')
        self.id_longitude = stream.sources.index('longitude')
        self.k = k
    def get_data(self, request=None):
        if request is not None: raise ValueError
        data = next(self.child_epoch_iterator)
        first_k = (numpy.array(at_least_k(self.k, data[self.id_latitude], False, False)[:self.k],
                               dtype=theano.config.floatX),
                   numpy.array(at_least_k(self.k, data[self.id_longitude], False, True)[:self.k],
                               dtype=theano.config.floatX))
        return data + first_k

class TaxiAddLastK(Transformer):
    def __init__(self, k, stream):
        super(TaxiAddLastK, self).__init__(stream)
        self.sources = stream.sources + ('last_k_latitude', 'last_k_longitude')
        self.id_latitude = stream.sources.index('latitude')
        self.id_longitude = stream.sources.index('longitude')
        self.k = k
    def get_data(self, request=None):
        if request is not None: raise ValueError
        data = next(self.child_epoch_iterator)
        last_k = (numpy.array(at_least_k(self.k, data[self.id_latitude], True, False)[-self.k:],
                            dtype=theano.config.floatX),
                  numpy.array(at_least_k(self.k, data[self.id_longitude], True, True)[-self.k:],
                              dtype=theano.config.floatX))
        return data + last_k

class TaxiAddDateTime(Transformer):
    def __init__(self, stream):
        super(TaxiAddDateTime, self).__init__(stream)
        self.sources = stream.sources + ('week_of_year', 'day_of_week', 'qhour_of_day')
        self.id_timestamp = stream.sources.index('timestamp')
    def get_data(self, request=None):
        if request is not None: raise ValueError
        data = next(self.child_epoch_iterator)
        ts = data[self.id_timestamp]
        date = datetime.datetime.utcfromtimestamp(ts)
        info = (date.isocalendar()[1] - 1, date.weekday(), date.hour * 4 + date.minute / 15)
        return data + info

class TaxiExcludeTrips(Transformer):
    def __init__(self, exclude_list, stream):
        super(TaxiExcludeTrips, self).__init__(stream)
        self.id_trip_id = stream.sources.index('trip_id')
        self.exclude = {v: True for v in exclude_list}
    def get_data(self, request=None):
        if request is not None: raise ValueError
        while True:
            data = next(self.child_epoch_iterator)
            if not data[self.id_trip_id] in self.exclude: break
        return data


