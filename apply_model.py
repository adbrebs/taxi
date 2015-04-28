import theano

from blocks.graph import ComputationGraph

class Apply(object):
    def __init__(self, outputs, return_vars, stream):
        if not isinstance(outputs, list):
            outputs = [outputs]
        if not isinstance(return_vars, list):
            return_vars = [return_vars]

        self.outputs = outputs
        self.return_vars = return_vars
        self.stream = stream

        cg = ComputationGraph(self.outputs)
        self.input_names = [i.name for i in cg.inputs]
        self.f = theano.function(inputs=cg.inputs, outputs=self.outputs)

    def __iter__(self):
        self.iterator = self.stream.get_epoch_iterator(as_dict=True)
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                return

            inputs = [batch[n] for n in self.input_names]
            outputs = self.f(*inputs)

            def find_retvar(name):
                for idx, ov in enumerate(self.outputs):
                    if ov.name == name:
                        return outputs[idx]

                if name in batch:
                    return batch[name]

                raise ValueError('Variable ' + name + ' neither in outputs or in batch variables.')

            yield {name: find_retvar(name) for name in self.return_vars}


