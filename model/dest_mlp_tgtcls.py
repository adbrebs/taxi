import numpy
import theano
from theano import tensor
from blocks.bricks import application, Softmax

import error
from model.mlp import FFMLP, Stream


class Model(FFMLP):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, output_layer=Softmax, **kwargs)
        self.classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX), name='classes')

    @application(outputs=['destination'])
    def predict(self, **kwargs):
        cls_probas = super(Model, self).predict(**kwargs)
        return tensor.dot(cls_probas, self.classes)

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @application(outputs=['cost'])
    def cost(self, **kwargs):
        y_hat = self.predict(**kwargs)
        y = tensor.concatenate((kwargs['destination_latitude'][:, None],
                                kwargs['destination_longitude'][:, None]), axis=1)

        return error.erdist(y_hat, y).mean()

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['destination_latitude', 'destination_longitude']
