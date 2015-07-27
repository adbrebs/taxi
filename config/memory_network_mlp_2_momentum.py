from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import Momentum

from blocks.bricks import Tanh

import data
from model.memory_network_mlp import Model, Stream

n_begin_end_pts = 5

dim_embeddings = [
    ('origin_call', data.origin_call_train_size, 10),
    ('origin_stand', data.stands_size, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 10),
]

embed_weights_init = IsotropicGaussian(0.001)

class MLPConfig(object):
    __slots__ = ('dim_input', 'dim_hidden', 'dim_output', 'weights_init', 'biases_init', 'embed_weights_init', 'dim_embeddings')

prefix_encoder = MLPConfig()
prefix_encoder.dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
prefix_encoder.dim_hidden = [100, 100]
prefix_encoder.weights_init = IsotropicGaussian(0.01)
prefix_encoder.biases_init = Constant(0.001)
prefix_encoder.embed_weights_init = embed_weights_init
prefix_encoder.dim_embeddings = dim_embeddings

candidate_encoder = MLPConfig()
candidate_encoder.dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
candidate_encoder.dim_hidden = [100, 100]
candidate_encoder.weights_init = IsotropicGaussian(0.01)
candidate_encoder.biases_init = Constant(0.001)
candidate_encoder.embed_weights_init = embed_weights_init
candidate_encoder.dim_embeddings = dim_embeddings

representation_size = 100
representation_activation = Tanh

normalize_representation = True

step_rule = Momentum(learning_rate=0.01, momentum=0.9)

batch_size = 1000
# batch_sort_size = 20

max_splits = 100

train_candidate_size = 5000
valid_candidate_size = 5000
test_candidate_size = 5000
