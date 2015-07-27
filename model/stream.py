from fuel.transformers import Batch, Padding, Mapping, SortMapping, Unpack, MultiProcessing, Filter
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, ShuffledExampleScheme

from theano import tensor

import data
from data import transformers
from data.hdf5 import TaxiDataset, TaxiStream


class StreamRec(object):
    def __init__(self, config):
        self.config = config

    def train(self, req_vars):
        stream = TaxiDataset('train', data.traintest_ds)

        if hasattr(self.config, 'use_cuts_for_training') and self.config.use_cuts_for_training:
            stream = DataStream(stream, iteration_scheme=TaxiTimeCutScheme())
        else:
            stream = DataStream(stream, iteration_scheme=ShuffledExampleScheme(stream.num_examples))

        if not data.tvt:
            valid = TaxiDataset(data.valid_set, data.valid_ds, sources=('trip_id',))
            valid_trips_ids = valid.get_data(None, slice(0, valid.num_examples))[0]
            stream = transformers.TaxiExcludeTrips(stream, valid_trips_ids)

        if hasattr(self.config, 'max_splits'):
            stream = transformers.TaxiGenerateSplits(stream, max_splits=self.config.max_splits)
        elif not data.tvt:
            stream = transformers.add_destination(stream)

        if hasattr(self.config, 'train_max_len'):
            idx = stream.sources.index('latitude')
            def max_len_filter(x):
                return len(x[idx]) <= self.config.train_max_len
            stream = Filter(stream, max_len_filter)

        stream = transformers.TaxiExcludeEmptyTrips(stream)
        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = transformers.balanced_batch(stream, key='latitude',
                                             batch_size=self.config.batch_size,
                                             batch_sort_size=self.config.batch_sort_size)
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        stream = transformers.Select(stream, req_vars)
        stream = MultiProcessing(stream)

        return stream

    def valid(self, req_vars):
        stream = TaxiStream(data.valid_set, data.valid_ds)

        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = transformers.balanced_batch(stream, key='latitude',
                                             batch_size=self.config.batch_size,
                                             batch_sort_size=self.config.batch_sort_size)
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        stream = transformers.Select(stream, req_vars)
        stream = MultiProcessing(stream)

        return stream

    def test(self, req_vars):
        stream = TaxiStream('test', data.traintest_ds)
        
        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.taxi_remove_test_only_clients(stream)

        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        stream = transformers.Select(stream, req_vars)
        return stream

    def inputs(self):
        return {'call_type': tensor.bvector('call_type'),
                'origin_call': tensor.ivector('origin_call'),
                'origin_stand': tensor.bvector('origin_stand'),
                'taxi_id': tensor.wvector('taxi_id'),
                'timestamp': tensor.ivector('timestamp'),
                'day_type': tensor.bvector('day_type'),
                'missing_data': tensor.bvector('missing_data'),
                'latitude': tensor.matrix('latitude'),
                'longitude': tensor.matrix('longitude'),
                'latitude_mask': tensor.matrix('latitude_mask'),
                'longitude_mask': tensor.matrix('longitude_mask'),
                'destination_latitude': tensor.vector('destination_latitude'),
                'destination_longitude': tensor.vector('destination_longitude'),
                'travel_time': tensor.ivector('travel_time'),
                'input_time': tensor.ivector('input_time'),
                'week_of_year': tensor.bvector('week_of_year'),
                'day_of_week': tensor.bvector('day_of_week'),
                'qhour_of_day': tensor.bvector('qhour_of_day')}

