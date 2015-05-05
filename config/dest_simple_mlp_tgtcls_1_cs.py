import cPickle

import data

import model.dest_simple_mlp_tgtcls as model

n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory
n_end_pts = 5

n_valid = 1000

with open(data.DATA_PATH + "/arrival-clusters.pkl") as f: tgtcls = cPickle.load(f)

dim_embeddings = [
    ('origin_call', data.n_train_clients+1, 10),
    ('origin_stand', data.n_stands+1, 10)
]

dim_input = n_begin_end_pts * 2 * 2 + sum(x for (_, _, x) in dim_embeddings)
dim_hidden = [500]
dim_output = tgtcls.shape[0]

learning_rate = 0.0001
momentum = 0.99
batch_size = 32
