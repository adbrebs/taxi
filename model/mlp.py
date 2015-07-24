from theano import tensor
import numpy

import fuel
import blocks

from fuel.transformers import Batch, MultiProcessing, Mapping, SortMapping, Unpack
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, ShuffledExampleScheme
from blocks.bricks import application, MLP, Rectifier, Initializable

import data
from data import transformers
from data.hdf5 import TaxiDataset, TaxiStream
from data.cut import TaxiTimeCutScheme
from model import ContextEmbedder

blocks.config.default_seed = 123
fuel.config.default_seed = 123

class FFMLP(Initializable):
    def __init__(self, config, output_layer=None, **kwargs):
        super(FFMLP, self).__init__(**kwargs)
        self.config = config

        self.context_embedder = ContextEmbedder(config)

        output_activation = [] if output_layer is None else [output_layer()]
        output_dim = [] if output_layer is None else [config.dim_output]
        self.mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden] + output_activation,
                       dims=[config.dim_input] + config.dim_hidden + output_dim)

        self.extremities = {'%s_k_%s' % (side, ['latitude', 'longitude'][axis]): axis for side in ['first', 'last'] for axis in [0, 1]}
        self.inputs = self.context_embedder.inputs + self.extremities.keys()
        self.children = [ self.context_embedder, self.mlp ]

    def _push_initialization_config(self):
        self.mlp.weights_init = self.config.mlp_weights_init
        self.mlp.biases_init = self.config.mlp_biases_init

    @application(outputs=['prediction'])
    def predict(self, **kwargs):
        embeddings = tuple(self.context_embedder.apply(**{k: kwargs[k] for k in self.context_embedder.inputs }))
        extremities = tuple((kwargs[k] - data.train_gps_mean[v]) / data.train_gps_std[v] for k, v in self.extremities.items())

        inputs = tensor.concatenate(extremities + embeddings, axis=1)
        outputs = self.mlp.apply(inputs)

        return outputs

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

class UniformGenerator(object):
    def __init__(self):
        self.rng = numpy.random.RandomState(123)
    def __call__(self, *args):
        return float(self.rng.uniform())

class Stream(object):
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

        stream = transformers.TaxiGenerateSplits(stream, max_splits=self.config.max_splits)

        if hasattr(self.config, 'shuffle_batch_size'):
            stream = transformers.Batch(stream, iteration_scheme=ConstantScheme(self.config.shuffle_batch_size))
            stream = Mapping(stream, SortMapping(key=UniformGenerator()))
            stream = Unpack(stream)

        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.taxi_add_first_last_len(stream, self.config.n_begin_end_pts)
        stream = transformers.Select(stream, tuple(req_vars))
        
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))

        stream = MultiProcessing(stream)

        return stream

    def valid(self, req_vars):
        stream = TaxiStream(data.valid_set, data.valid_ds)

        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.taxi_add_first_last_len(stream, self.config.n_begin_end_pts)
        stream = transformers.Select(stream, tuple(req_vars))
        return Batch(stream, iteration_scheme=ConstantScheme(1000))

    def test(self, req_vars):
        stream = TaxiStream('test', data.traintest_ds)
        
        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.taxi_add_first_last_len(stream, self.config.n_begin_end_pts)
        stream = transformers.taxi_remove_test_only_clients(stream)

        return Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))

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
                'qhour_of_day': tensor.bvector('qhour_of_day')}
