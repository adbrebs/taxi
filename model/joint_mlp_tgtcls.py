import numpy
import theano
from theano import tensor
from blocks import roles
from blocks.bricks import application, MLP, Rectifier, Softmax

import error
from model.mlp import FFMLP, Stream


class Model(FFMLP):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, **kwargs)
        
        self.dest_mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden_dest] + [Softmax()],
                       dims=[config.dim_hidden[-1]] + config.dim_hidden_dest + [config.dim_output_dest],
                       name='dest_mlp')
        self.time_mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden_time] + [Softmax()],
                       dims=[config.dim_hidden[-1]] + config.dim_hidden_time + [config.dim_output_time],
                       name='time_mlp')

        self.dest_classes = theano.shared(numpy.array(config.dest_tgtcls, dtype=theano.config.floatX), name='dest_classes')
        self.time_classes = theano.shared(numpy.array(config.time_tgtcls, dtype=theano.config.floatX), name='time_classes')

        self.inputs.append('input_time')
        self.children.extend([self.dest_mlp, self.time_mlp])

    def _push_initialization_config(self):
        super(Model, self)._push_initialization_config()
        for mlp in [self.dest_mlp, self.time_mlp]:
            mlp.weights_init = self.config.mlp_weights_init
            mlp.biases_init = self.config.mlp_biases_init

    @application(outputs=['destination', 'duration'])
    def predict(self, **kwargs):
        hidden = super(Model, self).predict(**kwargs)

        dest_cls_probas = self.dest_mlp.apply(hidden)
        dest_outputs = tensor.dot(dest_cls_probas, self.dest_classes)

        time_cls_probas = self.time_mlp.apply(hidden)
        time_outputs = kwargs['input_time'] + tensor.dot(time_cls_probas, self.time_classes)

        self.add_auxiliary_variable(dest_cls_probas, name='destination classes ponderations')
        self.add_auxiliary_variable(time_cls_probas, name='time classes ponderations')

        return (dest_outputs, time_outputs)

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @application(outputs=['cost'])
    def cost(self, **kwargs):
        (destination_hat, time_hat) = self.predict(**kwargs)

        destination = tensor.concatenate((kwargs['destination_latitude'][:, None],
                                          kwargs['destination_longitude'][:, None]), axis=1)
        time = kwargs['travel_time']

        destination_cost = error.erdist(destination_hat, destination).mean()
        time_cost = error.rmsle(time_hat.flatten(), time.flatten())

        self.add_auxiliary_variable(destination_cost, [roles.COST], 'destination_cost')
        self.add_auxiliary_variable(time_cost, [roles.COST], 'time_cost')

        return destination_cost + self.config.time_cost_factor * time_cost

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude', 'travel_time']
