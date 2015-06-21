from theano import tensor

from fuel.transformers import Batch, MultiProcessing
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, ShuffledExampleScheme
from blocks.bricks import application, MLP, Rectifier, Initializable, Identity

import error
import data
from data import transformers
from data.hdf5 import TaxiDataset, TaxiStream
from data.cut import TaxiTimeCutScheme
from model import ContextEmbedder


class Model(Initializable):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.config = config

        self.context_embedder = ContextEmbedder(config)
        self.mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden] + [Identity()],
                       dims=[config.dim_input] + config.dim_hidden + [config.dim_output])

        self.inputs = self.context_embedder.inputs # + self.extremities.keys()
        self.children = [ self.context_embedder, self.mlp ]

    def _push_initialization_config(self):
        self.mlp.weights_init = self.config.mlp_weights_init
        self.mlp.biases_init = self.config.mlp_biases_init

    @application(outputs=['destination'])
    def predict(self, **kwargs):
        embeddings = tuple(self.context_embedder.apply(**{k: kwargs[k] for k in self.context_embedder.inputs }))

        inputs = tensor.concatenate(embeddings, axis=1)
        outputs = self.mlp.apply(inputs)

        if self.config.output_mode == "destination":
            return data.train_gps_std * outputs + data.train_gps_mean
        elif self.config.dim_output == "clusters":
            return tensor.dot(outputs, self.classes)

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

        stream = TaxiDataset('train')

        if hasattr(self.config, 'use_cuts_for_training') and self.config.use_cuts_for_training:
            stream = DataStream(stream, iteration_scheme=TaxiTimeCutScheme())
        else:
            stream = DataStream(stream, iteration_scheme=ShuffledExampleScheme(stream.num_examples))

        stream = transformers.TaxiExcludeTrips(stream, valid_trips_ids)
        stream = transformers.TaxiGenerateSplits(stream, max_splits=self.config.max_splits)

        stream = transformers.taxi_add_datetime(stream)
        # stream = transformers.taxi_add_first_last_len(stream, self.config.n_begin_end_pts)
        stream = transformers.Select(stream, tuple(req_vars))
        
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))

        stream = MultiProcessing(stream)

        return stream

    def valid(self, req_vars):
        stream = TaxiStream(self.config.valid_set, 'valid.hdf5')

        stream = transformers.taxi_add_datetime(stream)
        # stream = transformers.taxi_add_first_last_len(stream, self.config.n_begin_end_pts)
        stream = transformers.Select(stream, tuple(req_vars))
        return Batch(stream, iteration_scheme=ConstantScheme(1000))

    def test(self, req_vars):
        stream = TaxiStream('test')
        
        stream = transformers.taxi_add_datetime(stream)
        # stream = transformers.taxi_add_first_last_len(stream, self.config.n_begin_end_pts)
        stream = transformers.taxi_remove_test_only_clients(stream)

        return Batch(stream, iteration_scheme=ConstantScheme(1))

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
