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

from blocks.bricks import MLP, Rectifier, Linear, Sigmoid, Identity
from blocks.bricks.lookup import LookupTable

from blocks.initialization import IsotropicGaussian, Constant
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
    config = importlib.import_module(sys.argv[1])


def setup_train_stream():
    # Load the training and test data
    train = H5PYDataset(data.H5DATA_PATH,
                        which_set='train',
                        subset=slice(0, data.dataset_size),
                        load_in_memory=True)
    train = DataStream(train, iteration_scheme=SequentialExampleScheme(data.dataset_size - config.n_valid))
    train = transformers.filter_out_trips(data.valid_trips, train)
    train = transformers.TaxiGenerateSplits(train)
    train = transformers.add_first_k(config.n_begin_end_pts, train)
    train = transformers.add_last_k(config.n_begin_end_pts, train)
    train = transformers.Select(train, ('origin_stand', 'origin_call', 'first_k_latitude',
                                        'last_k_latitude', 'first_k_longitude', 'last_k_longitude',
                                        'destination_latitude', 'destination_longitude'))
    train_stream = Batch(train, iteration_scheme=ConstantScheme(config.batch_size))

    return train_stream

def setup_valid_stream():
    valid = DataStream(data.valid_data)
    valid = transformers.add_first_k(config.n_begin_end_pts, valid)
    valid = transformers.add_last_k(config.n_begin_end_pts, valid)
    valid = transformers.Select(valid, ('origin_stand', 'origin_call', 'first_k_latitude',
                                        'last_k_latitude', 'first_k_longitude', 'last_k_longitude',
                                        'destination_latitude', 'destination_longitude'))
    valid_stream = Batch(valid, iteration_scheme=ConstantScheme(1000))
    
    return valid_stream

def setup_test_stream():
    test = data.test_data
    
    test = DataStream(test)
    test = transformers.add_first_k(config.n_begin_end_pts, test)
    test = transformers.add_last_k(config.n_begin_end_pts, test)
    test = transformers.Select(test, ('trip_id', 'origin_stand', 'origin_call', 'first_k_latitude',
                                      'last_k_latitude', 'first_k_longitude', 'last_k_longitude'))
    test_stream = Batch(test, iteration_scheme=ConstantScheme(1000))

    return test_stream


def main():
    # The input and the targets
    x_firstk_latitude = (tensor.matrix('first_k_latitude') - data.porto_center[0]) / data.data_std[0]
    x_firstk_longitude = (tensor.matrix('first_k_longitude') - data.porto_center[1]) / data.data_std[1]

    x_lastk_latitude = (tensor.matrix('last_k_latitude') - data.porto_center[0]) / data.data_std[0]
    x_lastk_longitude = (tensor.matrix('last_k_longitude') - data.porto_center[1]) / data.data_std[1]

    x_client = tensor.lvector('origin_call')
    x_stand = tensor.lvector('origin_stand')

    y = tensor.concatenate((tensor.vector('destination_latitude')[:, None],
                            tensor.vector('destination_longitude')[:, None]), axis=1)

    # x_firstk_latitude = theano.printing.Print("x_firstk_latitude")(x_firstk_latitude)
    # x_firstk_longitude = theano.printing.Print("x_firstk_longitude")(x_firstk_longitude)
    # x_lastk_latitude = theano.printing.Print("x_lastk_latitude")(x_lastk_latitude)
    # x_lastk_longitude = theano.printing.Print("x_lastk_longitude")(x_lastk_longitude)

    # Define the model
    client_embed_table = LookupTable(length=data.n_train_clients+1, dim=config.dim_embed, name='client_lookup')
    stand_embed_table = LookupTable(length=data.n_stands+1, dim=config.dim_embed, name='stand_lookup')
    mlp = MLP(activations=[Rectifier() for _ in config.dim_hidden] + [Identity()],
                       dims=[config.dim_input] + config.dim_hidden + [config.dim_output])

    # Create the Theano variables
    client_embed = client_embed_table.apply(x_client)
    stand_embed = stand_embed_table.apply(x_stand)
    inputs = tensor.concatenate([x_firstk_latitude, x_firstk_longitude,
                                 x_lastk_latitude, x_lastk_longitude,
                                 client_embed, stand_embed],
                                axis=1)
    # inputs = theano.printing.Print("inputs")(inputs)
    outputs = mlp.apply(inputs)

    # Normalize & Center
    # outputs = theano.printing.Print("normal_outputs")(outputs)
    outputs = data.data_std * outputs + data.porto_center

    # outputs = theano.printing.Print("outputs")(outputs)
    # y = theano.printing.Print("y")(y)

    outputs.name = 'outputs'

    # Calculate the cost
    cost = (outputs - y).norm(2, axis=1).mean()
    cost.name = 'cost'
    hcost = hdist.hdist(outputs, y).mean()
    hcost.name = 'hcost'

    # Initialization
    client_embed_table.weights_init = IsotropicGaussian(0.001)
    stand_embed_table.weights_init = IsotropicGaussian(0.001)
    mlp.weights_init = IsotropicGaussian(0.01)
    mlp.biases_init = Constant(0.001)

    client_embed_table.initialize()
    stand_embed_table.initialize()
    mlp.initialize()

    train_stream = setup_train_stream()
    valid_stream = setup_valid_stream()

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
                Dump('taxi_model', every_n_batches=1000),
                LoadFromDump('taxi_model'),
                FinishAfter(after_epoch=5)
                ]

    main_loop = MainLoop(
        model=Model([cost]),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions)
    main_loop.run()
    main_loop.profile.report()

    # Produce an output on the test data
    test_stream = setup_test_stream()

    outfile = open("test-output.csv", "w")
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

