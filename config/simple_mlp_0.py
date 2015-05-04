import model.simple_mlp as model

n_dow = 7       # number of division for dayofweek/dayofmonth/hourofday
n_dom = 31
n_hour = 24

n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory
n_end_pts = 5

n_valid = 1000

dim_embed = 10
dim_input = n_begin_end_pts * 2 * 2 + dim_embed + dim_embed
dim_hidden = [200, 100]
dim_output = 2

learning_rate = 0.0001
momentum = 0.99
batch_size = 32
