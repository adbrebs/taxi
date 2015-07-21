import numpy
import theano
from theano import tensor
from blocks.bricks.base import lazy
from blocks.bricks import Softmax

from model.rnn import RNN, Stream


class Model(RNN):
    @lazy()
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, output_dim=config.tgtcls.shape[0], **kwargs)
        self.classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX), name='classes')
        self.softmax = Softmax()
        self.children.append(self.softmax)

    def process_rto(self, rto):
        return tensor.dot(self.softmax.apply(rto), self.classes)
