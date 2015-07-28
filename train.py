#!/usr/bin/env python2

import importlib
import logging
import operator
import os
import sys
from functools import reduce

from theano import tensor

import blocks
import fuel

from blocks import roles
from blocks.algorithms import AdaDelta, CompositeRule, GradientDescent, RemoveNotFinite, StepRule, Momentum
from blocks.extensions import Printing, FinishAfter, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring

blocks.config.default_seed = 123
fuel.config.default_seed = 123

try:
    from blocks.extras.extensions.plot import Plot
    use_plot = True
except ImportError:
    use_plot = False
    
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.main_loop import MainLoop
from blocks.model import Model

from ext_saveload import SaveLoadParams
from ext_test import RunOnTest

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print >> sys.stderr, 'Usage: %s [--tvt | --largevalid] [--progress] config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[-1]
    config = importlib.import_module('.%s' % model_name, 'config')

    logger.info('# Configuration: %s' % config.__name__)
    for key in dir(config):
        if not key.startswith('__') and isinstance(getattr(config, key), (int, str, list, tuple)):
            logger.info('    %20s %s' % (key, str(getattr(config, key))))

    model = config.Model(config)
    model.initialize()

    stream = config.Stream(config)
    inputs = stream.inputs()
    req_vars = model.cost.inputs

    train_stream = stream.train(req_vars)
    valid_stream = stream.valid(req_vars)

    cost = model.cost(**inputs)
    cg = ComputationGraph(cost)
    monitored = set([cost] + VariableFilter(roles=[roles.COST])(cg.variables))

    valid_monitored = monitored
    if hasattr(model, 'valid_cost'):
        valid_cost = model.valid_cost(**inputs)
        valid_cg = ComputationGraph(valid_cost)
        valid_monitored = set([valid_cost] + VariableFilter(roles=[roles.COST])(valid_cg.variables))

    if hasattr(config, 'dropout') and config.dropout < 1.0:
        cg = apply_dropout(cg, config.dropout_inputs(cg), config.dropout)
    if hasattr(config, 'noise') and config.noise > 0.0:
        cg = apply_noise(cg, config.noise_inputs(cg), config.noise)
    cost = cg.outputs[0]
    cg = Model(cost)

    logger.info('# Parameter shapes:')
    parameters_size = 0
    for value in cg.parameters:
        logger.info('    %20s %s' % (value.get_value().shape, value.name))
        parameters_size += reduce(operator.mul, value.get_value().shape, 1)
    logger.info('Total number of parameters: %d in %d matrices' % (parameters_size, len(cg.parameters)))

    if hasattr(config, 'step_rule'):
        step_rule = config.step_rule
    else:
        step_rule = AdaDelta()

    logger.info("Fuel seed: %d" % fuel.config.default_seed)
    logger.info("Blocks seed: %d" % blocks.config.default_seed)

    params = cg.parameters
    algorithm = GradientDescent(
        cost=cost,
        step_rule=CompositeRule([
                RemoveNotFinite(),
                step_rule
            ]),
        parameters=params)
    
    plot_vars = [['valid_' + x.name for x in valid_monitored] +
                 ['train_' + x.name for x in valid_monitored]]
    logger.info('Plotted variables: %s' % str(plot_vars))

    dump_path = os.path.join('model_data', model_name) + '.pkl'
    logger.info('Dump path: %s' % dump_path)

    if hasattr(config, 'monitor_freq'):
        monitor_freq = config.monitor_freq
    else:
        monitor_freq = 10000

    extensions=[TrainingDataMonitoring(monitored, prefix='train', every_n_batches=monitor_freq),
                DataStreamMonitoring(valid_monitored, valid_stream,
                                     prefix='valid',
                                     every_n_batches=monitor_freq,
                                     after_epoch=False),
                Printing(every_n_batches=monitor_freq),
                FinishAfter(every_n_batches=10000000),

                SaveLoadParams(dump_path, cg,
                               before_training=True,        # before training -> load params
                               every_n_batches=monitor_freq,# every N batches -> save params
                               after_epoch=False,
                               after_training=True,         # after training -> save params
                               ),

                RunOnTest(model_name,
                          model,
                          stream,
                          every_n_batches=monitor_freq),
                ]

    if '--progress' in sys.argv:
        extensions.append(ProgressBar())
    
    if use_plot:
        extensions.append(Plot(model_name,
                               channels=plot_vars,
                               every_n_batches=500,
                               server_url='http://eos6:5006/'))

    main_loop = MainLoop(
        model=cg,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions)
    main_loop.run()
    main_loop.profile.report()
