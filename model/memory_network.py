
from theano import tensor

from fuel.transformers import Batch, MultiProcessing, Merge, Padding
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, ShuffledExampleScheme, SequentialExampleScheme
from blocks.bricks import application, MLP, Rectifier, Initializable, Softmax

import data
from data import transformers
from data.cut import TaxiTimeCutScheme
from data.hdf5 import TaxiDataset, TaxiStream
import error
from model import ContextEmbedder

class MemoryNetworkBase(Initializable):
    def __init__(self, config, prefix_encoder, candidate_encoder, **kwargs):
        super(MemoryNetworkBase, self).__init__(**kwargs)

        self.prefix_encoder = prefix_encoder
        self.candidate_encoder = candidate_encoder
        self.config = config

        self.softmax = Softmax()
        self.children = [ self.softmax, prefix_encoder, candidate_encoder ]

        self.inputs = self.prefix_encoder.apply.inputs \
                      + ['candidate_%s'%x for x in self.candidate_encoder.apply.inputs] \
                      + ['candidate_destination_latitude', 'candidate_destination_longitude']

    def candidate_destination(self, **kwargs):
        return tensor.concatenate(
                (tensor.shape_padright(kwargs['candidate_destination_latitude']),
                 tensor.shape_padright(kwargs['candidate_destination_longitude'])),
                axis=1)

    @application(outputs=['cost'])
    def cost(self, **kwargs):
        y_hat = self.predict(**kwargs)
        y = tensor.concatenate((kwargs['destination_latitude'][:, None],
                                kwargs['destination_longitude'][:, None]), axis=1)

        return error.erdist(y_hat, y).mean()

    @application(outputs=['destination'])
    def predict(self, **kwargs):
        prefix_representation = self.prefix_encoder.apply(**{ x: kwargs[x] for x in self.prefix_encoder.apply.inputs })
        candidate_representation = self.candidate_encoder.apply(**{ x: kwargs['candidate_'+x] for x in self.candidate_encoder.apply.inputs })

        if self.config.normalize_representation:
            prefix_representation = prefix_representation \
                    / tensor.sqrt((prefix_representation ** 2).sum(axis=1, keepdims=True))
            candidate_representation = candidate_representation \
                    / tensor.sqrt((candidate_representation ** 2).sum(axis=1, keepdims=True))

        similarity_score = tensor.dot(prefix_representation, candidate_representation.T)
        similarity = self.softmax.apply(similarity_score)

        return tensor.dot(similarity, self.candidate_destination(**kwargs))

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude']

class StreamBase(object):
    def __init__(self, config):
        self.config = config

        self.prefix_inputs = [
                ('call_type', tensor.bvector),
                ('origin_call', tensor.ivector),
                ('origin_stand', tensor.bvector),
                ('taxi_id', tensor.wvector),
                ('timestamp', tensor.ivector),
                ('day_type', tensor.bvector),
                ('missing_data', tensor.bvector),
                ('latitude', tensor.matrix),
                ('longitude', tensor.matrix),
                ('destination_latitude', tensor.vector),
                ('destination_longitude', tensor.vector),
                ('travel_time', tensor.ivector),
                ('input_time', tensor.ivector),
                ('week_of_year', tensor.bvector),
                ('day_of_week', tensor.bvector),
                ('qhour_of_day', tensor.bvector)
            ]
        self.candidate_inputs = self.prefix_inputs

    def inputs(self):
        prefix_inputs = { name: constructor(name)
                        for name, constructor in self.prefix_inputs }
        candidate_inputs = { 'candidate_'+name: constructor('candidate_'+name)
                             for name, constructor in self.candidate_inputs }
        return dict(prefix_inputs.items() + candidate_inputs.items())

    @property
    def valid_dataset(self):
        return TaxiDataset(data.valid_set, data.valid_ds)

    @property
    def valid_trips_ids(self):
        valid = TaxiDataset(data.valid_set, data.valid_ds, sources=('trip_id',))
        return valid.get_data(None, slice(0, valid.num_examples))[0]

    @property
    def train_dataset(self):
        return TaxiDataset('train', data.traintest_ds)

    @property
    def test_dataset(self):
        return TaxiDataset('test', data.traintest_ds)


class StreamSimple(StreamBase):
    def __init__(self, config):
        super(StreamSimple, self).__init__(config)

        self.prefix_inputs += [
                ('first_k_latitude', tensor.matrix),
                ('first_k_longitude', tensor.matrix),
                ('last_k_latitude', tensor.matrix),
                ('last_k_longitude', tensor.matrix),
        ]
        self.candidate_inputs = self.prefix_inputs

    def candidate_stream(self, n_candidates):
        candidate_stream = DataStream(self.train_dataset,
                                      iteration_scheme=ShuffledExampleScheme(self.train_dataset.num_examples))
        if not data.tvt:
            candidate_stream = transformers.TaxiExcludeTrips(candidate_stream, self.valid_trips_ids)
        candidate_stream = transformers.TaxiExcludeEmptyTrips(candidate_stream)
        candidate_stream = transformers.taxi_add_datetime(candidate_stream)
        candidate_stream = transformers.taxi_add_first_last_len(candidate_stream,
                                                                self.config.n_begin_end_pts)
        if not data.tvt:
            candidate_stream = transformers.add_destination(candidate_stream)

        return Batch(candidate_stream,
                     iteration_scheme=ConstantScheme(n_candidates))

    def train(self, req_vars):
        prefix_stream = DataStream(self.train_dataset,
                                   iteration_scheme=ShuffledExampleScheme(self.train_dataset.num_examples))

        if not data.tvt:
            prefix_stream = transformers.TaxiExcludeTrips(prefix_stream, self.valid_trips_ids)
        prefix_stream = transformers.TaxiExcludeEmptyTrips(prefix_stream)
        prefix_stream = transformers.TaxiGenerateSplits(prefix_stream,
                                                        max_splits=self.config.max_splits)
        prefix_stream = transformers.taxi_add_datetime(prefix_stream)
        prefix_stream = transformers.taxi_add_first_last_len(prefix_stream,
                                                             self.config.n_begin_end_pts)
        prefix_stream = Batch(prefix_stream,
                              iteration_scheme=ConstantScheme(self.config.batch_size))

        candidate_stream = self.candidate_stream(self.config.train_candidate_size)

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)
        stream = transformers.Select(stream, tuple(req_vars))
        stream = MultiProcessing(stream)
        return stream

    def valid(self, req_vars):
        prefix_stream = DataStream(
                           self.valid_dataset,
                           iteration_scheme=SequentialExampleScheme(self.valid_dataset.num_examples))
        prefix_stream = transformers.taxi_add_datetime(prefix_stream)
        prefix_stream = transformers.taxi_add_first_last_len(prefix_stream,
                                                             self.config.n_begin_end_pts)
        prefix_stream = Batch(prefix_stream,
                              iteration_scheme=ConstantScheme(self.config.batch_size))

        candidate_stream = self.candidate_stream(self.config.valid_candidate_size)

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)
        stream = transformers.Select(stream, tuple(req_vars))
        stream = MultiProcessing(stream)
        return stream

    def test(self, req_vars):
        prefix_stream = DataStream(
                           self.test_dataset,
                           iteration_scheme=SequentialExampleScheme(self.test_dataset.num_examples))
        prefix_stream = transformers.taxi_add_datetime(prefix_stream)
        prefix_stream = transformers.taxi_add_first_last_len(prefix_stream,
                                                             self.config.n_begin_end_pts)

        if not data.tvt:
            prefix_stream = transformers.taxi_remove_test_only_clients(prefix_stream)

        prefix_stream = Batch(prefix_stream,
                              iteration_scheme=ConstantScheme(self.config.batch_size))

        candidate_stream = self.candidate_stream(self.config.test_candidate_size)

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)
        stream = transformers.Select(stream, tuple(req_vars))
        stream = MultiProcessing(stream)
        return stream

class StreamRecurrent(StreamBase):
    def __init__(self, config):
        super(StreamRecurrent, self).__init__(config)

        self.prefix_inputs += [
                ('latitude_mask', tensor.matrix),
                ('longitude_mask', tensor.matrix),
        ]
        self.candidate_inputs = self.prefix_inputs

    def candidate_stream(self, n_candidates, sortmap=True):
        candidate_stream = DataStream(self.train_dataset,
                                      iteration_scheme=ShuffledExampleScheme(self.train_dataset.num_examples))
        if not data.tvt:
            candidate_stream = transformers.TaxiExcludeTrips(candidate_stream, self.valid_trips_ids)
        candidate_stream = transformers.TaxiExcludeEmptyTrips(candidate_stream)
        candidate_stream = transformers.taxi_add_datetime(candidate_stream)

        if not data.tvt:
            candidate_stream = transformers.add_destination(candidate_stream)

        if sortmap:
            candidate_stream = transformers.balanced_batch(candidate_stream,
                                                           key='latitude',
                                                           batch_size=n_candidates,
                                                           batch_sort_size=self.config.batch_sort_size)
        else:
            candidate_stream = Batch(candidate_stream,
                                     iteration_scheme=ConstantScheme(n_candidates))

        candidate_stream = Padding(candidate_stream,
                                   mask_sources=['latitude', 'longitude'])

        return candidate_stream

    def train(self, req_vars):
        prefix_stream = DataStream(self.train_dataset,
                                   iteration_scheme=ShuffledExampleScheme(self.train_dataset.num_examples))

        if not data.tvt:
            prefix_stream = transformers.TaxiExcludeTrips(prefix_stream, self.valid_trips_ids)
        prefix_stream = transformers.TaxiExcludeEmptyTrips(prefix_stream)
        prefix_stream = transformers.TaxiGenerateSplits(prefix_stream,
                                                        max_splits=self.config.max_splits)

        prefix_stream = transformers.taxi_add_datetime(prefix_stream)

        prefix_stream = transformers.balanced_batch(prefix_stream,
                                                    key='latitude',
                                                    batch_size=self.config.batch_size,
                                                    batch_sort_size=self.config.batch_sort_size)

        prefix_stream = Padding(prefix_stream, mask_sources=['latitude', 'longitude'])

        candidate_stream = self.candidate_stream(self.config.train_candidate_size)

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)

        stream = transformers.Select(stream, tuple(req_vars))
        # stream = MultiProcessing(stream)
        return stream

    def valid(self, req_vars):
        prefix_stream = DataStream(
                           self.valid_dataset,
                           iteration_scheme=SequentialExampleScheme(self.valid_dataset.num_examples))

        #prefix_stream = transformers.TaxiExcludeEmptyTrips(prefix_stream)

        prefix_stream = transformers.taxi_add_datetime(prefix_stream)

        prefix_stream = transformers.balanced_batch(prefix_stream,
                                                    key='latitude',
                                                    batch_size=self.config.batch_size,
                                                    batch_sort_size=self.config.batch_sort_size)

        prefix_stream = Padding(prefix_stream, mask_sources=['latitude', 'longitude'])

        candidate_stream = self.candidate_stream(self.config.valid_candidate_size)

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)

        stream = transformers.Select(stream, tuple(req_vars))
        # stream = MultiProcessing(stream)

        return stream

    def test(self, req_vars):
        prefix_stream = DataStream(
                           self.test_dataset,
                           iteration_scheme=SequentialExampleScheme(self.test_dataset.num_examples))

        prefix_stream = transformers.taxi_add_datetime(prefix_stream)
        if not data.tvt:
            prefix_stream = transformers.taxi_remove_test_only_clients(prefix_stream)

        prefix_stream = Batch(prefix_stream,
                              iteration_scheme=ConstantScheme(self.config.batch_size))
        prefix_stream = Padding(prefix_stream, mask_sources=['latitude', 'longitude'])

        candidate_stream = self.candidate_stream(self.config.test_candidate_size, False)

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)

        stream = transformers.Select(stream, tuple(req_vars))
        # stream = MultiProcessing(stream)

        return stream
