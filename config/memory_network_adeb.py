from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import AdaDelta, CompositeRule, GradientDescent, RemoveNotFinite, StepRule, Momentum

import data
from model.memory_network import Model, Stream


n_begin_end_pts = 5 # how many points we consider at the beginning and end of the known trajectory

dim_embeddings = [
    ('origin_call', data.origin_call_train_size, 10),
    ('origin_stand', data.stands_size, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 10),
]


class MLPConfig(object):
    __slots__ = ('dim_input', 'dim_hidden', 'dim_output', 'weights_init', 'biases_init')

prefix_encoder = MLPConfig()
prefix_encoder.dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
prefix_encoder.dim_hidden = [100, 100]
prefix_encoder.weights_init = IsotropicGaussian(0.001)
prefix_encoder.biases_init = Constant(0.0001)

candidate_encoder = MLPConfig()
candidate_encoder.dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
candidate_encoder.dim_hidden = [100, 100]
candidate_encoder.weights_init = IsotropicGaussian(0.001)
candidate_encoder.biases_init = Constant(0.0001)


embed_weights_init = IsotropicGaussian(0.001)

step_rule = Momentum(learning_rate=0.001, momentum=0.9)
batch_size = 32

max_splits = 1
num_cuts = 1000

train_candidate_size = 1000
valid_candidate_size = 10000

load_model = False
