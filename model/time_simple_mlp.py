from blocks.bricks import MLP, Rectifier, Linear, Sigmoid, Identity
from blocks.bricks.lookup import LookupTable

from blocks.initialization import IsotropicGaussian, Constant

from theano import tensor

import data
import error

class Model(object):
    def __init__(self, config):
        # The input and the targets
        x_firstk_latitude = (tensor.matrix('first_k_latitude') - data.porto_center[0]) / data.data_std[0]
        x_firstk_longitude = (tensor.matrix('first_k_longitude') - data.porto_center[1]) / data.data_std[1]

        x_lastk_latitude = (tensor.matrix('last_k_latitude') - data.porto_center[0]) / data.data_std[0]
        x_lastk_longitude = (tensor.matrix('last_k_longitude') - data.porto_center[1]) / data.data_std[1]

        input_list = [x_firstk_latitude, x_firstk_longitude, x_lastk_latitude, x_lastk_longitude]
        embed_tables = []

        self.require_inputs = ['first_k_latitude', 'first_k_longitude', 'last_k_latitude', 'last_k_longitude']

        for (varname, num, dim) in config.dim_embeddings:
            self.require_inputs.append(varname)
            vardata = tensor.lvector(varname)
            tbl = LookupTable(length=num, dim=dim, name='%s_lookup'%varname)
            embed_tables.append(tbl)
            input_list.append(tbl.apply(vardata))

        y = tensor.lvector('time')

        # Define the model
        mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden] + [Identity()],
                           dims=[config.dim_input] + config.dim_hidden + [config.dim_output])

        # Create the Theano variables
        inputs = tensor.concatenate(input_list, axis=1)
        # inputs = theano.printing.Print("inputs")(inputs)
        outputs = tensor.exp(mlp.apply(inputs) + 2)

        # outputs = theano.printing.Print("outputs")(outputs)
        # y = theano.printing.Print("y")(y)

        outputs.name = 'outputs'

        # Calculate the cost
        cost = error.rmsle(outputs.flatten(), y.flatten())
        cost.name = 'cost'

        # Initialization
        for tbl in embed_tables:
            tbl.weights_init = IsotropicGaussian(0.001)
        mlp.weights_init = IsotropicGaussian(0.01)
        mlp.biases_init = Constant(0.001)

        for tbl in embed_tables:
            tbl.initialize()
        mlp.initialize()

        self.cost = cost
        self.monitor = [cost]
        self.outputs = outputs
        self.pred_vars = ['time']
