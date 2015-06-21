
from theano import tensor

from fuel.transformers import Batch, MultiProcessing, Merge
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, ShuffledExampleScheme, SequentialExampleScheme
from blocks.bricks import application, MLP, Rectifier, Initializable, Softmax

import data
from data import transformers
from data.cut import TaxiTimeCutScheme
from data.hdf5 import TaxiDataset, TaxiStream
import error
from model import ContextEmbedder


class Model(Initializable):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.config = config

        self.context_embedder = ContextEmbedder(config)

        self.prefix_encoder = MLP(activations=[Rectifier() for _ in config.prefix_encoder.dim_hidden],
                                  dims=[config.prefix_encoder.dim_input] + config.prefix_encoder.dim_hidden,
                                  name='prefix_encoder')
        self.candidate_encoder = MLP(activations=[Rectifier() for _ in config.candidate_encoder.dim_hidden],
                                     dims=[config.candidate_encoder.dim_input] + config.candidate_encoder.dim_hidden,
                                     name='candidate_encoder')
        self.softmax = Softmax()

        self.prefix_extremities = {'%s_k_%s' % (side, ['latitude', 'longitude'][axis]): axis for side in ['first', 'last'] for axis in [0, 1]}
        self.candidate_extremities = {'candidate_%s_k_%s' % (side, ['latitude', 'longitude'][axis]): axis for side in ['first', 'last'] for axis in [0, 1]}

        self.inputs = self.context_embedder.inputs + ['candidate_%s'%k for k in self.context_embedder.inputs] + self.prefix_extremities.keys() + self.candidate_extremities.keys()
        self.children = [ self.context_embedder, self.prefix_encoder, self.candidate_encoder, self.softmax ]

    def _push_initialization_config(self):
        for (mlp, config) in [[self.prefix_encoder, self.config.prefix_encoder], [self.candidate_encoder, self.config.candidate_encoder]]:
            mlp.weights_init = config.weights_init
            mlp.biases_init = config.biases_init

    @application(outputs=['destination'])
    def predict(self, **kwargs):
        prefix_embeddings = tuple(self.context_embedder.apply(**{k: kwargs[k] for k in self.context_embedder.inputs }))
        prefix_extremities = tuple((kwargs[k] - data.train_gps_mean[v]) / data.train_gps_std[v] for k, v in self.prefix_extremities.items())
        prefix_inputs = tensor.concatenate(prefix_extremities + prefix_embeddings, axis=1)
        prefix_representation = self.prefix_encoder.apply(prefix_inputs)

        candidate_embeddings = tuple(self.context_embedder.apply(**{k: kwargs['candidate_%s'%k] for k in self.context_embedder.inputs }))
        candidate_extremities = tuple((kwargs[k] - data.train_gps_mean[v]) / data.train_gps_std[v] for k, v in self.candidate_extremities.items())
        candidate_inputs = tensor.concatenate(candidate_extremities + candidate_embeddings, axis=1)
        candidate_representation = self.candidate_encoder.apply(candidate_inputs)

        similarity_score = tensor.dot(prefix_representation, candidate_representation.T)
        similarity = self.softmax.apply(similarity_score)

        candidate_destination = tensor.concatenate(
                (tensor.shape_padright(kwargs['candidate_last_k_latitude'][:,-1]),
                 tensor.shape_padright(kwargs['candidate_last_k_longitude'][:,-1])),
                axis=1)

        return tensor.dot(similarity, candidate_destination)

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @application(outputs=['cost'])
    def cost(self, **kwargs):
        y_hat = self.predict(**kwargs)
        y = tensor.concatenate((kwargs['destination_latitude'][:, None],
                                kwargs['destination_longitude'][:, None]), axis=1)

        return error.erdist(y_hat, y).mean()

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude']

class Stream(object):
    def __init__(self, config):
        self.config = config

    def train(self, req_vars):
        valid = TaxiDataset(self.config.valid_set, 'valid.hdf5', sources=('trip_id',))
        valid_trips_ids = valid.get_data(None, slice(0, valid.num_examples))[0]

        dataset = TaxiDataset('train')

        prefix_stream = DataStream(dataset, iteration_scheme=TaxiTimeCutScheme())
        prefix_stream = transformers.TaxiExcludeTrips(prefix_stream, valid_trips_ids)
        prefix_stream = transformers.TaxiGenerateSplits(prefix_stream, max_splits=self.config.max_splits)
        prefix_stream = transformers.taxi_add_datetime(prefix_stream)
        prefix_stream = transformers.taxi_add_first_last_len(prefix_stream, self.config.n_begin_end_pts)
        prefix_stream = Batch(prefix_stream, iteration_scheme=ConstantScheme(self.config.batch_size))

        candidate_stream = DataStream(dataset, iteration_scheme=ShuffledExampleScheme(dataset.num_examples))
        candidate_stream = transformers.TaxiExcludeTrips(candidate_stream, valid_trips_ids)
        candidate_stream = transformers.TaxiExcludeEmptyTrips(candidate_stream)
        candidate_stream = transformers.taxi_add_datetime(candidate_stream)
        candidate_stream = transformers.taxi_add_first_last_len(candidate_stream, self.config.n_begin_end_pts)
        candidate_stream = Batch(candidate_stream, iteration_scheme=ConstantScheme(self.config.train_candidate_size))

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)
        stream = transformers.Select(stream, tuple(req_vars))
        stream = MultiProcessing(stream)
        return stream

    def valid(self, req_vars):
        valid_dataset = TaxiDataset(self.config.valid_set, 'valid.hdf5')
        train_dataset = TaxiDataset('train')
        valid_trips_ids = valid_dataset.get_data(None, slice(0, valid_dataset.num_examples))[valid_dataset.sources.index('trip_id')]

        prefix_stream = DataStream(valid_dataset, iteration_scheme=SequentialExampleScheme(valid_dataset.num_examples))
        prefix_stream = transformers.taxi_add_datetime(prefix_stream)
        prefix_stream = transformers.taxi_add_first_last_len(prefix_stream, self.config.n_begin_end_pts)
        prefix_stream = Batch(prefix_stream, iteration_scheme=ConstantScheme(self.config.batch_size))

        candidate_stream = DataStream(train_dataset, iteration_scheme=ShuffledExampleScheme(train_dataset.num_examples))
        candidate_stream = transformers.TaxiExcludeTrips(candidate_stream, valid_trips_ids)
        candidate_stream = transformers.TaxiExcludeEmptyTrips(candidate_stream)
        candidate_stream = transformers.taxi_add_datetime(candidate_stream)
        candidate_stream = transformers.taxi_add_first_last_len(candidate_stream, self.config.n_begin_end_pts)
        candidate_stream = Batch(candidate_stream, iteration_scheme=ConstantScheme(self.config.valid_candidate_size))

        sources = prefix_stream.sources + tuple('candidate_%s' % k for k in candidate_stream.sources)
        stream = Merge((prefix_stream, candidate_stream), sources)
        stream = transformers.Select(stream, tuple(req_vars))
        stream = MultiProcessing(stream)
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
                'destination_latitude': tensor.vector('destination_latitude'),
                'destination_longitude': tensor.vector('destination_longitude'),
                'travel_time': tensor.ivector('travel_time'),
                'first_k_latitude': tensor.matrix('first_k_latitude'),
                'first_k_longitude': tensor.matrix('first_k_longitude'),
                'last_k_latitude': tensor.matrix('last_k_latitude'),
                'last_k_longitude': tensor.matrix('last_k_longitude'),
                'input_time': tensor.ivector('input_time'),
                'week_of_year': tensor.bvector('week_of_year'),
                'day_of_week': tensor.bvector('day_of_week'),
                'qhour_of_day': tensor.bvector('qhour_of_day'),
                'candidate_call_type': tensor.bvector('candidate_call_type'),
                'candidate_origin_call': tensor.ivector('candidate_origin_call'),
                'candidate_origin_stand': tensor.bvector('candidate_origin_stand'),
                'candidate_taxi_id': tensor.wvector('candidate_taxi_id'),
                'candidate_timestamp': tensor.ivector('candidate_timestamp'),
                'candidate_day_type': tensor.bvector('candidate_day_type'),
                'candidate_missing_data': tensor.bvector('candidate_missing_data'),
                'candidate_latitude': tensor.matrix('candidate_latitude'),
                'candidate_longitude': tensor.matrix('candidate_longitude'),
                'candidate_destination_latitude': tensor.vector('candidate_destination_latitude'),
                'candidate_destination_longitude': tensor.vector('candidate_destination_longitude'),
                'candidate_travel_time': tensor.ivector('candidate_travel_time'),
                'candidate_first_k_latitude': tensor.matrix('candidate_first_k_latitude'),
                'candidate_first_k_longitude': tensor.matrix('candidate_first_k_longitude'),
                'candidate_last_k_latitude': tensor.matrix('candidate_last_k_latitude'),
                'candidate_last_k_longitude': tensor.matrix('candidate_last_k_longitude'),
                'candidate_input_time': tensor.ivector('candidate_input_time'),
                'candidate_week_of_year': tensor.bvector('candidate_week_of_year'),
                'candidate_day_of_week': tensor.bvector('candidate_day_of_week'),
                'candidate_qhour_of_day': tensor.bvector('candidate_qhour_of_day')}
