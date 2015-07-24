from blocks.initialization import IsotropicGaussian, Constant

import data
from model.mlp_emb import Model, Stream

use_cuts_for_training = True

dim_embeddings = [
    # ('origin_call', data.origin_call_train_size, 100),
    # ('origin_stand', data.stands_size, 100),
    # ('week_of_year', 52, 100),
    # ('day_of_week', 7, 100),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 1),
]

dim_input = sum(x for (_, _, x) in dim_embeddings)
dim_hidden = [10, 10]
output_mode = "destination"
dim_output = 2

embed_weights_init = IsotropicGaussian(0.01)
mlp_weights_init = IsotropicGaussian(0.01)
mlp_biases_init = IsotropicGaussian(0.001)

learning_rate = 0.001
momentum = 0.9
batch_size = 100

max_splits = 100
