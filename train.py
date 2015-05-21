#!/usr/bin/env python

import importlib
import logging
import operator
import os
import sys
from functools import reduce

from blocks import roles
from blocks.algorithms import AdaDelta, CompositeRule, GradientDescent, RemoveNotFinite
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.extensions.saveload import Dump, LoadFromDump
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.main_loop import MainLoop
from blocks.model import Model


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
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
    unmonitor = set()
    if hasattr(config, 'dropout') and config.dropout < 1.0:
        unmonitor.update(VariableFilter(roles=[roles.COST])(cg.variables))
        cg = apply_dropout(cg, config.dropout_inputs(cg), config.dropout)
    if hasattr(config, 'noise') and config.noise > 0.0:
        unmonitor.update(VariableFilter(roles=[roles.COST])(cg.variables))
        cg = apply_noise(cg, config.noise_inputs(cg), config.noise)
    cost = cg.outputs[0]
    cg = Model(cost)

    logger.info('# Parameter shapes:')
    parameters_size = 0
    for key, value in cg.get_params().iteritems():
        logger.info('    %20s %s' % (value.get_value().shape, key))
        parameters_size += reduce(operator.mul, value.get_value().shape, 1)
    logger.info('Total number of parameters: %d in %d matrices' % (parameters_size, len(cg.get_params())))

    params = cg.parameters
    algorithm = GradientDescent(
        cost=cost,
        step_rule=CompositeRule([
                RemoveNotFinite(),
                AdaDelta(),
                #Momentum(learning_rate=config.learning_rate, momentum=config.momentum),
            ]),
        params=params)
    
    monitored = set([cost] + VariableFilter(roles=[roles.COST])(cg.variables)) - unmonitor
    plot_vars = [['valid_' + x.name for x in monitored]]
    logger.info('Plotted variables: %s' % str(plot_vars))

    dump_path = os.path.join('model_data', model_name)
    logger.info('Dump path: %s' % dump_path)
    extensions=[TrainingDataMonitoring(monitored, prefix='train', every_n_batches=1000),
                DataStreamMonitoring(monitored, valid_stream,
                                     prefix='valid',
                                     every_n_batches=1000),
                Printing(every_n_batches=1000),
                Plot(model_name, channels=plot_vars, every_n_batches=500),
                Dump(dump_path, every_n_batches=5000),
                LoadFromDump(dump_path),
                #FinishAfter(after_n_batches=2),
                ]

    main_loop = MainLoop(
        model=cg,
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions)
    main_loop.run()
    main_loop.profile.report()
