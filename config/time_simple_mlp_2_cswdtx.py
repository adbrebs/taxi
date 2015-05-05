import model.time_simple_mlp as model

import data

n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory
n_end_pts = 5

n_valid = 1000

dim_embeddings = [
    ('origin_call', data.n_train_clients+1, 10),
    ('origin_stand', data.n_stands+1, 10),
    ('week_of_year', 52, 10),
    ('day_of_week', 7, 10),
    ('qhour_of_day', 24 * 4, 10),
    ('day_type', 3, 10),
    ('taxi_id', 448, 10),
]

dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
dim_hidden = [500, 100]
dim_output = 1

learning_rate = 0.00001
momentum = 0.99
batch_size = 32
