from blocks.bricks.base import lazy

from model.rnn import RNN, Stream
import data


class Model(RNN):
    @lazy()
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, **kwargs)

    def process_rto(self, rto):
        return (rto * data.train_gps_std) + data.train_gps_mean
