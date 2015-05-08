from blocks.bricks import MLP, Rectifier, Linear, Sigmoid, Identity, Softmax
from blocks.bricks.lookup import LookupTable

import numpy
import theano
from theano import tensor

import data
import error

class Model(object):
    def __init__(self, config):
        # The input and the targets
        x_firstk_latitude = (tensor.matrix('first_k_latitude') - data.train_gps_mean[0]) / data.train_gps_std[0]
        x_firstk_longitude = (tensor.matrix('first_k_longitude') - data.train_gps_mean[1]) / data.train_gps_std[1]

        x_lastk_latitude = (tensor.matrix('last_k_latitude') - data.train_gps_mean[0]) / data.train_gps_std[0]
        x_lastk_longitude = (tensor.matrix('last_k_longitude') - data.train_gps_mean[1]) / data.train_gps_std[1]

        x_input_time = tensor.lvector('input_time')

        input_list = [x_firstk_latitude, x_firstk_longitude, x_lastk_latitude, x_lastk_longitude]
        embed_tables = []

        self.require_inputs = ['first_k_latitude', 'first_k_longitude', 'last_k_latitude', 'last_k_longitude', 'input_time']

        for (varname, num, dim) in config.dim_embeddings:
            self.require_inputs.append(varname)
            vardata = tensor.lvector(varname)
            tbl = LookupTable(length=num, dim=dim, name='%s_lookup'%varname)
            embed_tables.append(tbl)
            input_list.append(tbl.apply(vardata))

        y_dest = tensor.concatenate((tensor.vector('destination_latitude')[:, None],
                                     tensor.vector('destination_longitude')[:, None]), axis=1)
        y_time = tensor.lvector('travel_time')

        # Define the model
        common_mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden],
                           dims=[config.dim_input] + config.dim_hidden)

        dest_mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden_dest] + [Softmax()],
                       dims=[config.dim_hidden[-1]] + config.dim_hidden_dest + [config.dim_output_dest],
                       name='dest_mlp')
        dest_classes = theano.shared(numpy.array(config.dest_tgtcls, dtype=theano.config.floatX), name='dest_classes')

        time_mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden_time] + [Softmax()],
                       dims=[config.dim_hidden[-1]] + config.dim_hidden_time + [config.dim_output_time],
                       name='time_mlp')
        time_classes = theano.shared(numpy.array(config.time_tgtcls, dtype=theano.config.floatX), name='time_classes')

        # Create the Theano variables
        inputs = tensor.concatenate(input_list, axis=1)
        # inputs = theano.printing.Print("inputs")(inputs)
        hidden = common_mlp.apply(inputs)

        dest_cls_probas = dest_mlp.apply(hidden)
        dest_outputs = tensor.dot(dest_cls_probas, dest_classes)
        dest_outputs.name = 'dest_outputs'

        time_cls_probas = time_mlp.apply(hidden)
        time_outputs = tensor.dot(time_cls_probas, time_classes) + x_input_time
        time_outputs.name = 'time_outputs'

        # Calculate the cost
        dest_cost = error.erdist(dest_outputs, y_dest).mean()
        dest_cost.name = 'dest_cost'
        dest_hcost = error.hdist(dest_outputs, y_dest).mean()
        dest_hcost.name = 'dest_hcost'

        time_cost = error.rmsle(time_outputs.flatten(), y_time.flatten())
        time_cost.name = 'time_cost'
        time_scost = config.time_cost_factor * time_cost
        time_scost.name = 'time_scost'

        cost = dest_cost + time_scost
        cost.name = 'cost'

        # Initialization
        for tbl in embed_tables:
            tbl.weights_init = config.embed_weights_init
            tbl.initialize()

        for mlp in [common_mlp, dest_mlp, time_mlp]:
            mlp.weights_init = config.mlp_weights_init
            mlp.biases_init = config.mlp_biases_init
            mlp.initialize()

        self.cost = cost
        self.monitor = [cost, dest_cost, dest_hcost, time_cost, time_scost]
        self.outputs = tensor.concatenate([dest_outputs, time_outputs[:, None]], axis=1)
        self.outputs.name = 'outputs'
        self.pred_vars = ['destination_longitude', 'destination_latitude', 'travel_time']

