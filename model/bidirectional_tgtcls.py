import numpy
import theano
from theano import tensor
from blocks.bricks.base import lazy
from blocks.bricks import Softmax

from model.bidirectional import BidiRNN, Stream


class Model(BidiRNN):
    @lazy()
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, output_dim=config.tgtcls.shape[0], **kwargs)

        self.classes = theano.shared(numpy.array(config.tgtcls, dtype=theano.config.floatX),
                                     name='classes')
        self.softmax = Softmax()
        self.children.append(self.softmax)

    def process_outputs(self, outputs):
        return tensor.dot(self.softmax.apply(outputs), self.classes)

