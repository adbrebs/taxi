from blocks.bricks.base import lazy

from model.bidirectional import BidiRNN, Stream
import data


class Model(BidiRNN):
    @lazy()
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, **kwargs)

    def process_outputs(self, outputs):
        return (outputs * data.train_gps_std) + data.train_gps_mean
