import os
import cPickle

from blocks.initialization import IsotropicGaussian, Constant
from blocks.algorithms import Momentum

import data
from model.dest_mlp_tgtcls import Model, Stream


n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory

with open(os.path.join(data.path, 'arrival-clusters.pkl')) as f: tgtcls = cPickle.load(f)

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
dim_output = tgtcls.shape[0]

embed_weights_init = IsotropicGaussian(0.01)
mlp_weights_init = IsotropicGaussian(0.1) 
mlp_biases_init = Constant(0.01)

step_rule = Momentum(learning_rate=0.001, momentum=0.99)

batch_size = 200

shuffle_batch_size = 5000

max_splits = 100
