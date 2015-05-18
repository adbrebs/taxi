from blocks.bricks import application, Initializable
from blocks.bricks.lookup import LookupTable


class ContextEmbedder(Initializable):
    def __init__(self, config, **kwargs):
        super(ContextEmbedder, self).__init__(**kwargs)
        self.dim_embeddings = config.dim_embeddings
        self.embed_weights_init = config.embed_weights_init

        self.inputs = [ name for (name, _, _) in self.dim_embeddings ]
        self.outputs = [ '%s_embedded' % name for name in self.inputs ]

        self.lookups = { name: LookupTable(name='%s_lookup' % name) for name in self.inputs }
        self.children = self.lookups.values()

    def _push_allocation_config(self):
        for (name, num, dim) in self.dim_embeddings:
            self.lookups[name].length = num
            self.lookups[name].dim = dim

    def _push_initialization_config(self):
        for name in self.inputs:
            self.lookups[name].weights_init = self.embed_weights_init

    @application
    def apply(self, **kwargs):
        return tuple(self.lookups[name].apply(kwargs[name]) for name in self.inputs)

    @apply.property('inputs')
    def apply_inputs(self):
        return self.inputs

    @apply.property('outputs')
    def apply_outputs(self):
        return self.outputs
