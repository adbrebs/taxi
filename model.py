import logging
import os
from argparse import ArgumentParser

from theano import tensor
from theano.ifelse import ifelse

from blocks.bricks import MLP, Rectifier, Linear
from blocks.bricks.lookup import LookupTable

from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model

from fuel.transformers import Batch
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme

from blocks.algorithms import GradientDescent, Scale
from blocks.graph import ComputationGraph
from blocks.main_loop import MainLoop
from blocks.extensions import Printing
from blocks.extensions.saveload import Dump, LoadFromDump
from blocks.extensions.monitoring import DataStreamMonitoring

import data

n_dow = 7       # number of division for dayofweek/dayofmonth/hourofday
n_dom = 31
n_hour = 24

n_clients = 57106
n_stands = 63
n_embed = n_clients + n_stands  # embeddings capturing local parameters

n_begin_pts = 5     # how many points we consider at the beginning and end of the known trajectory
n_end_pts = 5

dim_embed = 50
dim_hidden = 200

learning_rate = 0.1
batch_size = 32

def main():
    # The input and the targets
    x_firstk = tensor.matrix('first_k')
    x_lastk = tensor.matrix('last_k')
    x_client = tensor.lmatrix('client')
    y = tensor.vector('targets')

    # Define the model
    client_embed_table = LookupTable(length=n_clients, dim=dim_embed, name='lookup')
    hidden_layer = MLP(activations=[Rectifier()],
                       dims=[(n_begin_pts + n_end_pts) * 2 + dim_embed, dim_hidden])
    output_layer = Linear(input_dim=dim_hidden, output_dim=2)

    # Create the Theano variables

    client_embed = client_embed_table.apply(x_client).flatten(ndim=2)
    inputs = tensor.concatenate([x_firstk, x_lastk, client_embed], axis=1)
    hidden = hidden_layer.apply(inputs)
    outputs = output_layer.apply(hidden)

    # Calculate the cost
    cost = (outputs - y).norm(2, axis=1).mean()

    # Initialization
    client_embed_table.weights_init = IsotropicGaussian(0.001)
    hidden_layer.weights_init = IsotropicGaussian(0.01)
    hidden_layer.biases_init = Constant(0.001)
    output_layer.weights_init = IsotropicGaussian(0.001)
    output_layer.biases_init = Constant(0.001)

    client_embed_table.initialize()
    hidden_layer.initialize()
    output_layer.initialize()

    # Load the training and test data
    train = data.train_data
    stream = DataStream(train)
    train_stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))

    # valid = data.valid_data
    # stream = DataStream(valid)
    # valid_stream = Batch(stream, iteration_scheme=ConstantScheme(batch_size))
    valid_stream = train_stream

    # Training
    cg = ComputationGraph(cost)
    algorithm = GradientDescent(
        cost=cost,
        # step_rule=AdaDelta(decay_rate=0.5),
        step_rule=Scale(learning_rate=learning_rate),
        params=cg.parameters)

    extensions=[DataStreamMonitoring([cost], valid_stream,
                                     prefix='valid',
                                     every_n_batches=100),
                    Printing(every_n_batches=100),
                    Dump('ngram_blocks_model', every_n_batches=100),
                    LoadFromDump('ngram_blocks_model')]

    main_loop = MainLoop(
        model=Model([cost]),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=extensions)
    main_loop.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

