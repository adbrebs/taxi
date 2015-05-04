import numpy 

import theano
from theano import tensor

from blocks.bricks import MLP, Rectifier, Linear, Sigmoid, Identity, Softmax
from blocks.bricks.lookup import LookupTable

from blocks.initialization import IsotropicGaussian, Constant

import data
import hdist

class Model(object):
    def __init__(self, config):
        # The input and the targets
        x_firstk_latitude = (tensor.matrix('first_k_latitude') - data.porto_center[0]) / data.data_std[0]
        x_firstk_longitude = (tensor.matrix('first_k_longitude') - data.porto_center[1]) / data.data_std[1]

        x_lastk_latitude = (tensor.matrix('last_k_latitude') - data.porto_center[0]) / data.data_std[0]
        x_lastk_longitude = (tensor.matrix('last_k_longitude') - data.porto_center[1]) / data.data_std[1]

        x_client = tensor.lvector('origin_call')
        x_stand = tensor.lvector('origin_stand')

        y = tensor.concatenate((tensor.vector('destination_latitude')[:, None],
                                tensor.vector('destination_longitude')[:, None]), axis=1)

        # Define the model
        client_embed_table = LookupTable(length=data.n_train_clients+1, dim=config.dim_embed, name='client_lookup')
        stand_embed_table = LookupTable(length=data.n_stands+1, dim=config.dim_embed, name='stand_lookup')
        mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden] + [Softmax()],
                           dims=[config.dim_input] + config.dim_hidden + [config.dim_output])
        classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX), name='classes')

        # Create the Theano variables
        client_embed = client_embed_table.apply(x_client)
        stand_embed = stand_embed_table.apply(x_stand)
        inputs = tensor.concatenate([x_firstk_latitude, x_firstk_longitude,
                                     x_lastk_latitude, x_lastk_longitude,
                                     client_embed, stand_embed],
                                    axis=1)
        # inputs = theano.printing.Print("inputs")(inputs)
        cls_probas = mlp.apply(inputs)
        outputs = tensor.dot(cls_probas, classes)

        # outputs = theano.printing.Print("outputs")(outputs)
        # y = theano.printing.Print("y")(y)

        outputs.name = 'outputs'

        # Calculate the cost
        cost = hdist.erdist(outputs, y).mean()
        cost.name = 'cost'
        hcost = hdist.hdist(outputs, y).mean()
        hcost.name = 'hcost'

        # Initialization
        client_embed_table.weights_init = IsotropicGaussian(0.001)
        stand_embed_table.weights_init = IsotropicGaussian(0.001)
        mlp.weights_init = IsotropicGaussian(0.01)
        mlp.biases_init = Constant(0.001)

        client_embed_table.initialize()
        stand_embed_table.initialize()
        mlp.initialize()

        self.cost = cost
        self.hcost = hcost
        self.outputs = outputs
