import numpy
import theano
from theano import tensor
from blocks.bricks import application, Softmax

import error
from model.mlp import FFMLP, Stream


class Model(FFMLP):
    def __init__(self, config, **kwargs):
        super(Model, self, output_layer=Softmax).__init__(config, **kwargs)
        self.classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX), name='classes')
        self.inputs.append('input_time')

    @application(outputs=['duration'])
    def predict(self, **kwargs):
        cls_probas = super(Model, self).predict(**kwargs)
        return kwargs['input_time'] + tensor.dot(cls_probas, self.classes)

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @application(outputs=['cost'])
    def cost(self, **kwargs):
        y_hat = self.predict(**kwargs)
        y = kwargs['travel_time']
        return error.rmsle(y_hat.flatten(), y.flatten())

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['travel_time']
