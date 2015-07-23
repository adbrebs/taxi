
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


class Model(MemoryNetworkBase):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

        self.prefix_encoder = MLP(activations=[Rectifier() for _ in config.prefix_encoder.dim_hidden]
                                              + [config.representation_activation()],
                                  dims=[config.prefix_encoder.dim_input]
                                       + config.prefix_encoder.dim_hidden
                                       + [config.representation_size],
                                  name='prefix_encoder')

        self.candidate_encoder = MLP(
                    activations=[Rectifier() for _ in config.candidate_encoder.dim_hidden]
                                + [config.representation_activation()],
                    dims=[config.candidate_encoder.dim_input]
                         + config.candidate_encoder.dim_hidden
                         + [config.representation_size],
                    name='candidate_encoder')
        self.softmax = Softmax()

        self.prefix_extremities = {'%s_k_%s' % (side, ['latitude', 'longitude'][axis]): axis 
                                   for side in ['first', 'last'] for axis in [0, 1]}
        self.candidate_extremities = {'candidate_%s_k_%s' % (side, axname): axis
                                      for side in ['first', 'last']
                                      for axis, axname in enumerate(['latitude', 'longitude'])}

        self.inputs = self.context_embedder.inputs \
                      + ['candidate_%s'%k for k in self.context_embedder.inputs] \
                      + self.prefix_extremities.keys() + self.candidate_extremities.keys()
        self.children = [ self.context_embedder,
                          self.prefix_encoder,
                          self.candidate_encoder,
                          self.softmax ]

    def _push_initialization_config(self):
        for (mlp, config) in [[self.prefix_encoder, self.config.prefix_encoder],
                              [self.candidate_encoder, self.config.candidate_encoder]]:
            mlp.weights_init = config.weights_init
            mlp.biases_init = config.biases_init

    @application(outputs=['destination'])
    def predict(self, **kwargs):
        prefix_embeddings = tuple(self.context_embedder.apply(
                                **{k: kwargs[k] for k in self.context_embedder.inputs }))
        prefix_extremities = tuple((kwargs[k] - data.train_gps_mean[v]) / data.train_gps_std[v]
                                   for k, v in self.prefix_extremities.items())
        prefix_inputs = tensor.concatenate(prefix_extremities + prefix_embeddings, axis=1)
        prefix_representation = self.prefix_encoder.apply(prefix_inputs)
        if self.config.normalize_representation:
            prefix_representation = prefix_representation \
                    / tensor.sqrt((prefix_representation ** 2).sum(axis=1, keepdims=True))

        candidate_embeddings = tuple(self.context_embedder.apply(**{k: kwargs['candidate_%s'%k]
                                     for k in self.context_embedder.inputs }))
        candidate_extremities = tuple((kwargs[k] - data.train_gps_mean[v]) / data.train_gps_std[v]
                                      for k, v in self.candidate_extremities.items())
        candidate_inputs = tensor.concatenate(candidate_extremities + candidate_embeddings, axis=1)
        candidate_representation = self.candidate_encoder.apply(candidate_inputs)
        if self.config.normalize_representation:
            candidate_representation = candidate_representation \
                    / tensor.sqrt((candidate_representation ** 2).sum(axis=1, keepdims=True))

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
