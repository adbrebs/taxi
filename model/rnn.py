from theano import tensor

from blocks.bricks import application, MLP, Initializable, Tanh
from blocks.bricks.base import lazy
from blocks.bricks.recurrent import LSTM, recurrent
from blocks.utils import shared_floatx_zeros

from fuel.transformers import Batch, Padding
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, ShuffledExampleScheme

from model import ContextEmbedder
import data
from data import transformers
from data.hdf5 import TaxiDataset, TaxiStream
import error


from model.stream import StreamRec as Stream

class RNN(Initializable):
    @lazy()
    def __init__(self, config, rec_input_len=2, output_dim=2, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.config = config

        self.pre_context_embedder = ContextEmbedder(config.pre_embedder, name='pre_context_embedder')
        self.post_context_embedder = ContextEmbedder(config.post_embedder, name='post_context_embedder')

        in1 = rec_input_len + sum(x[2] for x in config.pre_embedder.dim_embeddings)
        self.input_to_rec = MLP(activations=[Tanh()], dims=[in1, config.hidden_state_dim], name='input_to_rec')

        self.rec = LSTM(
                dim = config.hidden_state_dim,
                name = 'recurrent'
            )

        in2 = config.hidden_state_dim + sum(x[2] for x in config.post_embedder.dim_embeddings)
        self.rec_to_output = MLP(activations=[Tanh()], dims=[in2, output_dim], name='rec_to_output')

        self.sequences = ['latitude', 'latitude_mask', 'longitude']
        self.context = self.pre_context_embedder.inputs + self.post_context_embedder.inputs
        self.inputs = self.sequences + self.context
        self.children = [ self.pre_context_embedder, self.post_context_embedder, self.input_to_rec, self.rec, self.rec_to_output ]

        self.initial_state_ = shared_floatx_zeros((config.hidden_state_dim,),
                name="initial_state")
        self.initial_cells = shared_floatx_zeros((config.hidden_state_dim,),
                name="initial_cells")

    def _push_initialization_config(self):
        for mlp in [self.input_to_rec, self.rec_to_output]:
            mlp.weights_init = self.config.weights_init
            mlp.biases_init = self.config.biases_init
        self.rec.weights_init = self.config.weights_init

    def get_dim(self, name):
        return self.rec.get_dim(name)

    def process_rto(self, rto):
        return rto

    def rec_input(self, latitude, longitude, **kwargs):
        return (tensor.shape_padright(latitude), tensor.shape_padright(longitude))

    @recurrent(states=['states', 'cells'], outputs=['destination', 'states', 'cells'])
    def predict_all(self, **kwargs):
        pre_emb = tuple(self.pre_context_embedder.apply(**kwargs))

        itr_in = tensor.concatenate(pre_emb + self.rec_input(**kwargs), axis=1)
        itr = self.input_to_rec.apply(itr_in)
        itr = itr.repeat(4, axis=1)
        (next_states, next_cells) = self.rec.apply(itr, kwargs['states'], kwargs['cells'], mask=kwargs['latitude_mask'], iterate=False)

        post_emb = tuple(self.post_context_embedder.apply(**kwargs))
        rto = self.rec_to_output.apply(tensor.concatenate(post_emb + (next_states,), axis=1))

        rto = self.process_rto(rto)
        return (rto, next_states, next_cells)

    @predict_all.property('sequences')
    def predict_all_sequences(self):
        return self.sequences

    @application(outputs=predict_all.states)
    def initial_states(self, *args, **kwargs):
        return self.rec.initial_states(*args, **kwargs)

    @predict_all.property('contexts')
    def predict_all_context(self):
        return self.context

    def before_predict_all(self, kwargs):
        kwargs['latitude'] = (kwargs['latitude'].T - data.train_gps_mean[0]) / data.train_gps_std[0]
        kwargs['longitude'] = (kwargs['longitude'].T - data.train_gps_mean[1]) / data.train_gps_std[1]
        kwargs['latitude_mask'] = kwargs['latitude_mask'].T

    @application(outputs=['destination'])
    def predict(self, **kwargs):
        self.before_predict_all(kwargs)
        res = self.predict_all(**kwargs)[0]

        last_id = tensor.cast(kwargs['latitude_mask'].sum(axis=0) - 1, dtype='int64')
        return res[last_id, tensor.arange(kwargs['latitude_mask'].shape[1])]

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @application(outputs=['cost_matrix'])
    def cost_matrix(self, **kwargs):
        self.before_predict_all(kwargs)

        res = self.predict_all(**kwargs)[0]
        target = tensor.concatenate(
                    (kwargs['destination_latitude'].dimshuffle('x', 0, 'x'),
                     kwargs['destination_longitude'].dimshuffle('x', 0, 'x')),
                axis=2)
        target = target.repeat(kwargs['latitude'].shape[0], axis=0)
        ce = error.erdist(target.reshape((-1, 2)), res.reshape((-1, 2)))
        ce = ce.reshape(kwargs['latitude'].shape)
        return ce * kwargs['latitude_mask']

    @cost_matrix.property('inputs')
    def cost_matrix_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude']

    @application(outputs=['cost'])
    def cost(self, latitude_mask, **kwargs):
        return self.cost_matrix(latitude_mask=latitude_mask, **kwargs).sum() / latitude_mask.sum()

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude']

    @application(outputs=['cost'])
    def valid_cost(self, **kwargs):
        last_id = tensor.cast(kwargs['latitude_mask'].sum(axis=1) - 1, dtype='int64')
        return self.cost_matrix(**kwargs)[last_id, tensor.arange(kwargs['latitude_mask'].shape[0])].mean()

    @valid_cost.property('inputs')
    def valid_cost_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude']


