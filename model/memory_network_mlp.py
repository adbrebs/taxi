
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

from memory_network import StreamSimple as Stream
from memory_network import MemoryNetworkBase

class MLPEncoder(Initializable):
    def __init__(self, config, output_dim, activation, **kwargs):
        super(MLPEncoder, self).__init__(**kwargs)

        self.config = config
        self.context_embedder = ContextEmbedder(self.config)

        self.encoder_mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden]
                                           + [activation()],
                               dims=[config.dim_input]
                                    + config.dim_hidden
                                    + [output_dim],
                               name='encoder')

        self.extremities = {'%s_k_%s' % (side, ['latitude', 'longitude'][axis]): axis 
                             for side in ['first', 'last'] for axis in [0, 1]}

        self.children = [ self.context_embedder,
                          self.encoder_mlp ]

    def _push_initialization_config(self):
        for brick in [self.context_embedder, self.encoder_mlp]:
            brick.weights_init = self.config.weights_init
            brick.biases_init = self.config.biases_init

    @application
    def apply(self, **kwargs):
        embeddings = tuple(self.context_embedder.apply(
                           **{k: kwargs[k] for k in self.context_embedder.inputs }))
        extremities = tuple((kwargs[k] - data.train_gps_mean[v]) / data.train_gps_std[v]
                            for k, v in self.extremities.items())
        inputs = tensor.concatenate(extremities + embeddings, axis=1)

        return self.encoder_mlp.apply(inputs)

    @apply.property('inputs')
    def apply_inputs(self):
        return self.context_embedder.inputs + self.extremities.keys()


class Model(MemoryNetworkBase):
    def __init__(self, config, **kwargs):
        prefix_encoder = MLPEncoder(config.prefix_encoder,
                                    config.representation_size,
                                    config.representation_activation,
                                    name='prefix_encoder')

        candidate_encoder = MLPEncoder(config.candidate_encoder,
                                      config.representation_size,
                                      config.representation_activation,
                                      name='candidate_encoder')

        super(Model, self).__init__(config, prefix_encoder, candidate_encoder, **kwargs)
