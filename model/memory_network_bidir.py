from theano import tensor

from toolz import merge

from blocks.bricks import application, MLP, Rectifier, Initializable, Softmax, Linear
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import Bidirectional, LSTM

import data
from data import transformers
from data.cut import TaxiTimeCutScheme
from data.hdf5 import TaxiDataset, TaxiStream
import error
from model import ContextEmbedder

from memory_network import StreamRecurrent as Stream
from memory_network import MemoryNetworkBase
from bidirectional import SegregatedBidirectional


class RecurrentEncoder(Initializable):
    def __init__(self, config, output_dim, activation, **kwargs):
        super(RecurrentEncoder, self).__init__(**kwargs)

        self.config = config
        self.context_embedder = ContextEmbedder(config)

        self.rec = SegregatedBidirectional(LSTM(dim=config.rec_state_dim, name='encoder_recurrent'))

        self.fwd_fork = Fork([name for name in self.rec.prototype.apply.sequences if name!='mask'],
                             prototype=Linear(), name='fwd_fork')
        self.bkwd_fork = Fork([name for name in self.rec.prototype.apply.sequences if name!='mask'],
                              prototype=Linear(), name='bkwd_fork')

        rto_in = config.rec_state_dim * 2 + sum(x[2] for x in config.dim_embeddings)
        self.rec_to_output = MLP(
                    activations=[Rectifier() for _ in config.dim_hidden] + [activation],
                    dims=[rto_in] + config.dim_hidden + [output_dim],
                    name='encoder_rto')

        self.children = [self.context_embedder, self.rec, self.fwd_fork, self.bkwd_fork, self.rec_to_output]

        self.rec_inputs = ['latitude', 'longitude', 'latitude_mask']
        self.inputs = self.context_embedder.inputs + self.rec_inputs

    def _push_allocation_config(self):
        for i, fork in enumerate([self.fwd_fork, self.bkwd_fork]):
            fork.input_dim = 2
            fork.output_dims = [ self.rec.children[i].get_dim(name)
                                 for name in fork.output_names ]

    def _push_initialization_config(self):
        for brick in self.children:
            brick.weights_init = self.config.weights_init
            brick.biases_init = self.config.biases_init

    @application
    def apply(self, latitude, longitude, latitude_mask, **kwargs):
        latitude = (latitude.T - data.train_gps_mean[0]) / data.train_gps_std[0]
        longitude = (longitude.T - data.train_gps_mean[1]) / data.train_gps_std[1]
        latitude_mask = latitude_mask.T

        rec_in = tensor.concatenate((latitude[:, :, None], longitude[:, :, None]),
                                    axis=2)
        path = self.rec.apply(merge(self.fwd_fork.apply(rec_in, as_dict=True),
                                    {'mask': latitude_mask}),
                              merge(self.bkwd_fork.apply(rec_in, as_dict=True),
                                    {'mask': latitude_mask}))[0]

        last_id = tensor.cast(latitude_mask.sum(axis=0) - 1, dtype='int64')
        
        path_representation = (path[0][:, -self.config.rec_state_dim:],
                path[last_id - 1, tensor.arange(last_id.shape[0])]
                    [:, :self.config.rec_state_dim])

        embeddings = tuple(self.context_embedder.apply(
                            **{k: kwargs[k] for k in self.context_embedder.inputs }))

        inputs = tensor.concatenate(path_representation + embeddings, axis=1)
        outputs = self.rec_to_output.apply(inputs)

        return outputs

    @apply.property('inputs')
    def apply_inputs(self):
        return self.inputs


class Model(MemoryNetworkBase):
    def __init__(self, config, **kwargs):

        # Build prefix encoder : recurrent then MLP
        prefix_encoder = RecurrentEncoder(config.prefix_encoder,
                                          config.representation_size,
                                          config.representation_activation(),
                                          name='prefix_encoder')

        # Build candidate encoder
        candidate_encoder = RecurrentEncoder(config.candidate_encoder,
                                             config.representation_size,
                                             config.representation_activation(),
                                             name='candidate_encoder')

        # And... that's it!
        super(Model, self).__init__(config, prefix_encoder, candidate_encoder, **kwargs)
