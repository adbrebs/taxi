from blocks.initialization import IsotropicGaussian, Constant

from blocks.bricks import Tanh

import data
from model.memory_network_bidir import Model, Stream


dim_embeddings = [
    ('origin_call', data.origin_call_train_size, 10),
    ('origin_stand', data.stands_size, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 10),
]

embed_weights_init = IsotropicGaussian(0.001)


class RNNConfig(object):
    __slots__ = ('rec_state_dim', 'dim_embeddings', 'embed_weights_init',
                 'dim_hidden', 'weights_init', 'biases_init')

prefix_encoder = RNNConfig()
prefix_encoder.dim_embeddings = dim_embeddings
prefix_encoder.embed_weights_init = embed_weights_init
prefix_encoder.rec_state_dim = 100
prefix_encoder.dim_hidden = [100, 100]
prefix_encoder.weights_init = IsotropicGaussian(0.01)
prefix_encoder.biases_init = Constant(0.001)

candidate_encoder = RNNConfig()
candidate_encoder.dim_embeddings = dim_embeddings
candidate_encoder.embed_weights_init = embed_weights_init
candidate_encoder.rec_state_dim = 100
candidate_encoder.dim_hidden = [100, 100]
candidate_encoder.weights_init = IsotropicGaussian(0.01)
candidate_encoder.biases_init = Constant(0.001)

representation_size = 100
representation_activation = Tanh

normalize_representation = True


batch_size = 100
batch_sort_size = 20

max_splits = 100

train_candidate_size = 1000
valid_candidate_size = 1000
test_candidate_size = 1000
