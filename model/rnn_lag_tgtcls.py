import numpy
import theano
from theano import tensor
from blocks.bricks.base import lazy
from blocks.bricks import Softmax

from model.rnn import RNN, Stream


class Model(RNN):
    @lazy()
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, rec_input_len=4, output_dim=config.tgtcls.shape[0], **kwargs)
        self.classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX), name='classes')
        self.softmax = Softmax()
        self.sequences.extend(['latitude_lag', 'longitude_lag'])
        self.children.append(self.softmax)

    def before_predict_all(self, kwargs):
        super(Model, self).before_predict_all(kwargs)
        kwargs['latitude_lag'] = tensor.extra_ops.repeat(kwargs['latitude'], 2, axis=0)
        kwargs['longitude_lag'] = tensor.extra_ops.repeat(kwargs['longitude'], 2, axis=0)

    def process_rto(self, rto):
        return tensor.dot(self.softmax.apply(rto), self.classes)

    def rec_input(self, latitude, longitude, latitude_lag, longitude_lag, **kwargs):
        return (tensor.shape_padright(latitude),
                tensor.shape_padright(longitude),
                tensor.shape_padright(latitude_lag),
                tensor.shape_padright(longitude_lag))
