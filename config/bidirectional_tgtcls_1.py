import os
import cPickle

from blocks.initialization import IsotropicGaussian, Constant

import data
from model.bidirectional_tgtcls import Model, Stream


with open(os.path.join(data.path, 'arrival-clusters.pkl')) as f: tgtcls = cPickle.load(f)

dim_embeddings = [
    ('origin_call', data.origin_call_train_size, 10),
    ('origin_stand', data.stands_size, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('taxi_id', 448, 10),
]

hidden_state_dim = 100

dim_hidden = [500, 500]

embed_weights_init = IsotropicGaussian(0.01)
fork_weights_init = IsotropicGaussian(0.1) 
fork_biases_init = Constant(0.01)
rec_weights_init = IsotropicGaussian(0.1) 
mlp_weights_init = IsotropicGaussian(0.1) 
mlp_biases_init = Constant(0.01)

batch_size = 20
batch_sort_size = 20

valid_set = 'cuts/large_valid'
max_splits = 100
