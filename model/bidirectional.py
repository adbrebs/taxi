from theano import tensor

from blocks.bricks import application, MLP, Initializable, Linear, Rectifier, Identity
from blocks.bricks.base import lazy
from blocks.bricks.recurrent import Bidirectional, LSTM
from blocks.utils import shared_floatx_zeros
from blocks.bricks.parallel import Fork

from fuel.transformers import Batch, Padding, Mapping, SortMapping, Unpack, MultiProcessing
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, ShuffledExampleScheme

from model import ContextEmbedder
import data
from data import transformers
from data.hdf5 import TaxiDataset, TaxiStream
import error


class BidiRNN(Initializable):
    @lazy()
    def __init__(self, config, output_dim=2, **kwargs):
        super(BidiRNN, self).__init__(**kwargs)
        self.config = config

        self.context_embedder = ContextEmbedder(config)
        
        self.rec = Bidirectional(LSTM(dim = config.hidden_state_dim, name = 'recurrent'))

        self.fork = Fork([name for name in self.rec.prototype.apply.sequences if name!='mask'], prototype=Linear())

        rto_in = config.hidden_state_dim * 2 + sum(x[2] for x in config.dim_embeddings)
        self.rec_to_output = MLP(activations=[Rectifier() for _ in config.dim_hidden] + [Identity()], dims=[rto_in] + config.dim_hidden + [output_dim])

        self.sequences = ['latitude', 'latitude_mask', 'longitude']
        self.inputs = self.sequences + self.context_embedder.inputs

        self.children = [ self.context_embedder, self.fork, self.rec, self.rec_to_output ]

    def _push_allocation_config(self):
        self.fork.input_dim = 2
        self.fork.output_dims = [ self.rec.children[0].get_dim(name) for name in self.fork.output_names ]
        self.fork.weights_init = self.config.fork_weights_init
        self.fork.biases_init = self.config.fork_biases_init
        self.rec.weights_init = self.config.rec_weights_init
        self.rec_to_output.weights_init = self.config.mlp_weights_init
        self.rec_to_output.biases_init = self.config.mlp_biases_init

    def process_outputs(self, outputs):
        return outputs

    @application(outputs=['destination'])
    def predict(self, latitude, longitude, latitude_mask, **kwargs):
        latitude = (latitude.T - data.train_gps_mean[0]) / data.train_gps_std[0]
        longitude = (longitude.T - data.train_gps_mean[1]) / data.train_gps_std[1]
        latitude_mask = latitude_mask.T

        latitude = tensor.shape_padright(latitude)
        longitude = tensor.shape_padright(longitude)
        rec_in = tensor.concatenate((latitude, longitude), axis=2)

        last_id = tensor.cast(latitude_mask.sum(axis=0) - 1, dtype='int64')
        path = self.rec.apply(self.fork.apply(rec_in), mask=latitude_mask)[0]
        path_representation = (path[0][:, -self.config.hidden_state_dim:],
                path[last_id - 1, tensor.arange(latitude_mask.shape[1])][:, :self.config.hidden_state_dim])

        embeddings = tuple(self.context_embedder.apply(**{k: kwargs[k] for k in self.context_embedder.inputs }))

        inputs = tensor.concatenate(path_representation + embeddings, axis=1)
        outputs = self.rec_to_output.apply(inputs)

        return self.process_outputs(outputs)

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

class UniformGenerator(object):
    def __init__(self):
        self.rng = numpy.random.RandomState(123)
    def __call__(self, *args):
        return float(self.rng.uniform())

class Stream(object):
    def __init__(self, config):
        self.config = config

    def train(self, req_vars):
        valid = TaxiDataset(self.config.valid_set, 'valid.hdf5', sources=('trip_id',))
        valid_trips_ids = valid.get_data(None, slice(0, valid.num_examples))[0]

        stream = TaxiDataset('train')

        if hasattr(self.config, 'use_cuts_for_training') and self.config.use_cuts_for_training:
            stream = DataStream(stream, iteration_scheme=TaxiTimeCutScheme())
        else:
            stream = DataStream(stream, iteration_scheme=ShuffledExampleScheme(stream.num_examples))

        stream = transformers.TaxiExcludeTrips(stream, valid_trips_ids)
        stream = transformers.TaxiGenerateSplits(stream, max_splits=self.config.max_splits)

        if hasattr(self.config, 'shuffle_batch_size'):
            stream = transformers.Batch(stream, iteration_scheme=ConstantScheme(self.config.shuffle_batch_size))
            stream = Mapping(stream, SortMapping(key=UniformGenerator()))
            stream = Unpack(stream)

        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = transformers.balanced_batch(stream, key='latitude', batch_size=self.config.batch_size, batch_sort_size=self.config.batch_sort_size)
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        
        stream = transformers.Select(stream, req_vars)
        stream = MultiProcessing(stream)

        return stream

    def valid(self, req_vars):
        stream = TaxiStream(self.config.valid_set, 'valid.hdf5')

        stream = transformers.taxi_add_datetime(stream)
        stream = transformers.Select(stream, tuple(v for v in req_vars if not v.endswith('_mask')))

        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        stream = Padding(stream, mask_sources=['latitude', 'longitude'])
        stream = transformers.Select(stream, req_vars)
        return stream

    def test(self, req_vars):
        stream = TaxiStream('test')
        
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
