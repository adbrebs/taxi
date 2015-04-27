import logging
import os
from argparse import ArgumentParser

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

from fuel.transformers import Batch
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme

from blocks.algorithms import GradientDescent, Scale, AdaDelta
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.extensions.saveload import Dump, LoadFromDump
from blocks.extensions.monitoring import DataStreamMonitoring

import data
import transformers
import hdist

n_dow = 7       # number of division for dayofweek/dayofmonth/hourofday
n_dom = 31
n_hour = 24

n_clients = 57105
n_stands = 63

n_begin_end_pts = 5     # how many points we consider at the beginning and end of the known trajectory
n_end_pts = 5

dim_embed = 50
dim_hidden = 200

learning_rate = 0.01
batch_size = 32

def main():
    # The input and the targets
    x_firstk = tensor.matrix('first_k')
    x_lastk = tensor.matrix('last_k')
    x_client = tensor.lvector('origin_call')
    x_stand = tensor.lvector('origin_stand')
    y = tensor.matrix('destination')

    # Define the model
    client_embed_table = LookupTable(length=n_clients+1, dim=dim_embed, name='client_lookup')
    stand_embed_table = LookupTable(length=n_stands+1, dim=dim_embed, name='stand_lookup')
    hidden_layer = MLP(activations=[Rectifier()],
                       dims=[n_begin_end_pts * 2 * 2 + dim_embed + dim_embed, dim_hidden])
    output_layer = Linear(input_dim=dim_hidden, output_dim=2)

    # Create the Theano variables

    client_embed = client_embed_table.apply(x_client).flatten(ndim=2)
    stand_embed = stand_embed_table.apply(x_stand).flatten(ndim=2)
    inputs = tensor.concatenate([x_firstk, x_lastk,
                                 client_embed.zeros_like(), stand_embed.zeros_like()],
                                axis=1)
    # inputs = theano.printing.Print("inputs")(inputs)
    hidden = hidden_layer.apply(inputs)
    # hidden = theano.printing.Print("hidden")(hidden)
    outputs = output_layer.apply(hidden)

    # Calculate the cost
    # cost = (outputs - y).norm(2, axis=1).mean()
    # outputs = numpy.array([[ -8.621953, 41.162142]], dtype='float32') + 0 * outputs
    cost = (outputs - y).norm(2, axis=1).mean()
    cost.name = 'cost'
    hcost = hdist.hdist(outputs, y).mean()
    hcost.name = 'hcost'

    # Initialization
    client_embed_table.weights_init = IsotropicGaussian(0.001)
    stand_embed_table.weights_init = IsotropicGaussian(0.001)
    hidden_layer.weights_init = IsotropicGaussian(0.01)
    hidden_layer.biases_init = Constant(0.001)
    output_layer.weights_init = IsotropicGaussian(0.001)
    output_layer.biases_init = Constant(0.001)

    client_embed_table.initialize()
    stand_embed_table.initialize()
    hidden_layer.initialize()
    output_layer.initialize()

    # Load the training and test data
    train = data.train_data
    train = DataStream(train)
    train = transformers.add_first_k(n_begin_end_pts, train)
    train = transformers.add_random_k(n_begin_end_pts, train)
    train = transformers.add_destination(train)
    train = transformers.Select(train, ('origin_stand', 'origin_call', 'first_k', 'last_k', 'destination'))
    train_stream = Batch(train, iteration_scheme=ConstantScheme(batch_size))

    valid = data.valid_data
    valid = DataStream(valid)
    valid = transformers.add_first_k(n_begin_end_pts, valid)
    valid = transformers.add_last_k(n_begin_end_pts, valid)
    valid = transformers.concat_destination_xy(valid)
    valid = transformers.Select(valid, ('origin_stand', 'origin_call', 'first_k', 'last_k', 'destination'))
    valid_stream = Batch(valid, iteration_scheme=ConstantScheme(batch_size))


    # Training
    cg = ComputationGraph(cost)
    params = VariableFilter(bricks=[Linear])(cg.parameters)
    algorithm = GradientDescent(
        cost=cost,
        step_rule=AdaDelta(decay_rate=0.5),
        # step_rule=Scale(learning_rate=learning_rate),
        params=params)

    extensions=[DataStreamMonitoring([cost, hcost], valid_stream,
                                     prefix='valid',
                                     every_n_batches=1000),
                Printing(every_n_batches=1000),
                # Dump('taxi_model', every_n_batches=100),
                # LoadFromDump('taxi_model'),
                ]

    main_loop = MainLoop(
        model=Model([cost]),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions)
    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

