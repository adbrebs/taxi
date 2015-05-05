import logging
import os
import sys
import importlib
from argparse import ArgumentParser

import csv

import numpy

import theano
from theano import printing
from theano import tensor
from theano.ifelse import ifelse

from blocks.filter import VariableFilter

from blocks.model import Model

from fuel.datasets.hdf5 import H5PYDataset
from fuel.transformers import Batch
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme, SequentialExampleScheme

from blocks.algorithms import GradientDescent, Scale, AdaDelta, Momentum
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.extensions import Printing, FinishAfter
from blocks.extensions.saveload import Dump, LoadFromDump, Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring

import data
import transformers
import hdist
import apply_model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module(model_name)


def setup_train_stream(req_vars):
    # Load the training and test data
    train = H5PYDataset(data.H5DATA_PATH,
                        which_set='train',
                        subset=slice(0, data.dataset_size),
                        load_in_memory=True)
    train = DataStream(train, iteration_scheme=SequentialExampleScheme(data.dataset_size - config.n_valid))

    train = transformers.TaxiExcludeTrips(data.valid_trips, train)
    train = transformers.TaxiGenerateSplits(train, max_splits=100)

    train = transformers.TaxiAddDateTime(train)
    train = transformers.TaxiAddFirstK(config.n_begin_end_pts, train)
    train = transformers.TaxiAddLastK(config.n_begin_end_pts, train)
    train = transformers.Select(train, tuple(req_vars))

    train_stream = Batch(train, iteration_scheme=ConstantScheme(config.batch_size))

    return train_stream

def setup_valid_stream(req_vars):
    valid = DataStream(data.valid_data)

    valid = transformers.TaxiAddDateTime(valid)
    valid = transformers.TaxiAddFirstK(config.n_begin_end_pts, valid)
    valid = transformers.TaxiAddLastK(config.n_begin_end_pts, valid)
    valid = transformers.Select(valid, tuple(req_vars))

    valid_stream = Batch(valid, iteration_scheme=ConstantScheme(1000))
    
    return valid_stream

def setup_test_stream(req_vars):
    test = DataStream(data.test_data)
    
    test = transformers.TaxiAddDateTime(test)
    test = transformers.TaxiAddFirstK(config.n_begin_end_pts, test)
    test = transformers.TaxiAddLastK(config.n_begin_end_pts, test)
    test = transformers.Select(test, tuple(req_vars))

    test_stream = Batch(test, iteration_scheme=ConstantScheme(1000))

    return test_stream


def main():
    model = config.model.Model(config)

    cost = model.cost
    hcost = model.hcost
    outputs = model.outputs

    req_vars = model.require_inputs + [ 'destination_latitude', 'destination_longitude' ]
    req_vars_test = model.require_inputs + [ 'trip_id' ]

    train_stream = setup_train_stream(req_vars)
    valid_stream = setup_valid_stream(req_vars)

    # Training
    cg = ComputationGraph(cost)
    params = cg.parameters # VariableFilter(bricks=[Linear])(cg.parameters) 
    algorithm = GradientDescent(
        cost=cost,
        # step_rule=AdaDelta(decay_rate=0.5),
        step_rule=Momentum(learning_rate=config.learning_rate, momentum=config.momentum),
        params=params)

    extensions=[DataStreamMonitoring([cost, hcost], valid_stream,
                                     prefix='valid',
                                     every_n_batches=1000),
                Printing(every_n_batches=1000),
                # Checkpoint('model.pkl', every_n_batches=100),
                Dump('model_data/' + model_name, every_n_batches=1000),
                LoadFromDump('model_data/' + model_name),
                FinishAfter(after_epoch=42),
                ]

    main_loop = MainLoop(
        model=Model([cost]),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions)
    main_loop.run()
    main_loop.profile.report()

    # Produce an output on the test data
    test_stream = setup_test_stream(req_vars_test)

    outfile = open("output/test-output-%s.csv" % model_name, "w")
    outcsv = csv.writer(outfile)
    outcsv.writerow(["TRIP_ID", "LATITUDE", "LONGITUDE"])
    for out in apply_model.Apply(outputs=outputs, stream=test_stream, return_vars=['trip_id', 'outputs']):
        dest = out['outputs']
        for i, trip in enumerate(out['trip_id']):
            outcsv.writerow([trip, repr(dest[i, 0]), repr(dest[i, 1])])
    outfile.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

