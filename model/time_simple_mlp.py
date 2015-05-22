from blocks.bricks import application, Identity

import error
from model.mlp import FFMLP, Stream


class Model(FFMLP):
    def __init__(self, config, **kwargs):
        super(Model, self).__init__(config, output_layer=Identity, **kwargs)
        self.inputs.append('input_time')

    @application(outputs=['duration'])
    def predict(self, **kwargs):
        outputs = super(Model, self).predict(**kwargs).flatten()
        if hasattr(self.config, 'exp_base'):
            outputs = self.config.exp_base ** outputs
        return kwargs['input_time'] + outputs

    @predict.property('inputs')
    def predict_inputs(self):
        return self.inputs

    @application(outputs=['cost'])
    def cost(self, **kwargs):
        y_hat = self.predict(**kwargs)
        y = kwargs['travel_time']
        return error.rmsle(y_hat, y)

    @cost.property('inputs')
    def cost_inputs(self):
        return self.inputs + ['travel_time']
