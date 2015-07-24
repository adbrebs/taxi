import os
import cPickle

from blocks.initialization import IsotropicGaussian, Constant

import data
from model.joint_mlp_tgtcls import Model, Stream


n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory

with open(os.path.join(data.path, 'arrival-clusters.pkl')) as f:
    dest_tgtcls = cPickle.load(f)

# generate target classes for time prediction as a Fibonacci sequence
time_tgtcls = [1, 2]
for i in range(22):
    time_tgtcls.append(time_tgtcls[-1] + time_tgtcls[-2])

dim_embeddings = [
    ('origin_call', data.origin_call_size, 10),
    ('origin_stand', data.stands_size, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 10),
    ('taxi_id', 448, 10),
]

# Common network part
dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
dim_hidden = [500]

# Destination prediction part
dim_hidden_dest = []
dim_output_dest = len(dest_tgtcls)

# Time prediction part
dim_hidden_time = []
dim_output_time = len(time_tgtcls)

# Cost ratio between distance cost and time cost
time_cost_factor = 4

embed_weights_init = IsotropicGaussian(0.001)
mlp_weights_init = IsotropicGaussian(0.01)
mlp_biases_init = Constant(0.001)

learning_rate = 0.0001
momentum = 0.99
batch_size = 200

max_splits = 100
