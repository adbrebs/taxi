import model.time_simple_mlp as model

import data

n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory
n_end_pts = 5

n_valid = 1000

dim_embeddings = [
]

dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
dim_hidden = [200]
dim_output = 1

learning_rate = 0.00001
momentum = 0.99
batch_size = 32
