import cPickle
import logging

from blocks.extensions import SimpleExtension

logger = logging.getLogger(__name__)

class SaveLoadParams(SimpleExtension):
    def __init__(self, path, model, **kwargs):
        super(SaveLoadParams, self).__init__(**kwargs)

        self.path = path
        self.model = model
    
    def do_save(self):
        with open(self.path, 'w') as f:
            logger.info('Saving parameters to %s...'%self.path)
            cPickle.dump(self.model.get_parameter_values(), f, protocol=cPickle.HIGHEST_PROTOCOL)
            logger.info('Done saving.')
    
    def do_load(self):
        try:
            with open(self.path, 'r') as f:
                logger.info('Loading parameters from %s...'%self.path)
                self.model.set_parameter_values(cPickle.load(f))
        except IOError:
            pass

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            self.do_load()
        else:
            self.do_save()
