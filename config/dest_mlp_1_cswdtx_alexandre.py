import os
import cPickle

from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import Momentum

import data
from model.dest_mlp import Model, Stream


n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory

dim_embeddings = [
    ('origin_call', data.origin_call_train_size, 10),
    ('origin_stand', data.stands_size, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 10),
    ('taxi_id', 448, 10),
]

dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
dim_hidden = [500]
dim_output = 2

embed_weights_init = IsotropicGaussian(0.01)
mlp_weights_init = IsotropicGaussian(0.1) 
mlp_biases_init = Constant(0.01)

step_rule = Momentum(learning_rate=0.001, momentum=0.9)

batch_size = 200

max_splits = 100
