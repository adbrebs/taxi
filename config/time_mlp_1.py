from blocks.initialization import IsotropicGaussian, Constant

import data
from model.time_mlp import Model, Stream


n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory

dim_embeddings = [
]

dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
dim_hidden = [200]
dim_output = 1

embed_weights_init = IsotropicGaussian(0.001)
mlp_weights_init = IsotropicGaussian(0.01)
mlp_biases_init = Constant(0.001)

exp_base = 1.5

learning_rate = 0.00001
momentum = 0.99
batch_size = 32

max_splits = 100
