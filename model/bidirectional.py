from theano import tensor

from toolz import merge

from blocks.bricks import application, MLP, Initializable, Linear, Rectifier, Identity
from blocks.bricks.base import lazy
from blocks.bricks.recurrent import Bidirectional, LSTM
from blocks.utils import shared_floatx_zeros
from blocks.bricks.parallel import Fork

from model import ContextEmbedder
import error

import data

from model.stream import StreamRec as Stream

class SegregatedBidirectional(Bidirectional):
    @application
    def apply(self, forward_dict, backward_dict):
        """Applies forward and backward networks and concatenates outputs."""

        forward = self.children[0].apply(as_list=True, **forward_dict)
        backward = [x[::-1] for x in

        self.children[1].apply(reverse=True, as_list=True,
                **backward_dict)]

        return [tensor.concatenate([f, b], axis=2)
                for f, b in zip(forward, backward)]

class BidiRNN(Initializable):
    @lazy()
    def __init__(self, config, output_dim=2, **kwargs):
        super(BidiRNN, self).__init__(**kwargs)
        self.config = config

        self.context_embedder = ContextEmbedder(config)
        
        act = config.rec_activation() if hasattr(config, 'rec_activation') else None
        self.rec = SegregatedBidirectional(LSTM(dim=config.hidden_state_dim, activation=act, name='recurrent'))

        self.fwd_fork = Fork([name for name in self.rec.prototype.apply.sequences if name!='mask'],
                             prototype=Linear(), name='fwd_fork')
        self.bkwd_fork = Fork([name for name in self.rec.prototype.apply.sequences if name!='mask'],
                              prototype=Linear(), name='bkwd_fork')

        rto_in = config.hidden_state_dim * 2 + sum(x[2] for x in config.dim_embeddings)
        self.rec_to_output = MLP(activations=[Rectifier() for _ in config.dim_hidden] + [Identity()], 
                                 dims=[rto_in] + config.dim_hidden + [output_dim])

        self.sequences = ['latitude', 'latitude_mask', 'longitude']
        self.inputs = self.sequences + self.context_embedder.inputs

        self.children = [ self.context_embedder, self.fwd_fork, self.bkwd_fork,
                          self.rec, self.rec_to_output ]

    def _push_allocation_config(self):
        for i, fork in enumerate([self.fwd_fork, self.bkwd_fork]):
            fork.input_dim = 2
            fork.output_dims = [ self.rec.children[i].get_dim(name)
                                 for name in fork.output_names ]

    def _push_initialization_config(self):
        for brick in [self.fwd_fork, self.bkwd_fork, self.rec, self.rec_to_output]:
            brick.weights_init = self.config.weights_init
            brick.biases_init = self.config.biases_init

    def process_outputs(self, outputs):
        pass # must be implemented in child class

    @application(outputs=['destination'])
    def predict(self, latitude, longitude, latitude_mask, **kwargs):
        latitude = (latitude.T - data.train_gps_mean[0]) / data.train_gps_std[0]
        longitude = (longitude.T - data.train_gps_mean[1]) / data.train_gps_std[1]
        latitude_mask = latitude_mask.T

        rec_in = tensor.concatenate((latitude[:, :, None], longitude[:, :, None]), axis=2)

        last_id = tensor.cast(latitude_mask.sum(axis=0) - 1, dtype='int64')

        path = self.rec.apply(merge(self.fwd_fork.apply(rec_in, as_dict=True),
                                    {'mask': latitude_mask}),
                              merge(self.bkwd_fork.apply(rec_in, as_dict=True),
                                    {'mask': latitude_mask}))[0]

        path_representation = (path[0][:, -self.config.hidden_state_dim:],
                               path[last_id - 1, tensor.arange(latitude_mask.shape[1])]
                                   [:, :self.config.hidden_state_dim])

        embeddings = tuple(self.context_embedder.apply(
                        **{k: kwargs[k] for k in self.context_embedder.inputs }))

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


