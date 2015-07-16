import cPickle
import numpy as np
import matplotlib.pyplot as plt

from fuel.schemes import ConstantScheme
from fuel.streams import DataStream

import data
from data.hdf5 import TaxiDataset, TaxiStream


def compute_number_coordinates():
    stream = TaxiDataset('train').get_example_stream()
    train_it = stream.get_epoch_iterator()

    # Count the number of coordinates
    n_coordinates = 0
    for ride in train_it:
        n_coordinates += len(ride[-1])
    print n_coordinates

    return n_coordinates


def extract_coordinates(n_coordinates=None):
    """Extract coordinates from the dataset and store them in a numpy array"""

    if n_coordinates is None:
        n_coordinates = compute_number_coordinates()

    dataset = TaxiDataset('train')
    stream = DataStream(dataset, iteration_scheme=ConstantScheme(1, dataset.num_examples))

    coordinates = np.zeros((n_coordinates, 2), dtype="float32")
    train_it = stream.get_epoch_iterator()

    c = 0
    for ride in train_it:
        for point in zip(ride[2], ride[3]):
            coordinates[c] = point
            c += 1
    print c

    cPickle.dump(coordinates, open(data.path + "/coordinates_array.pkl", "wb"))


def draw_map(coordinates, xrg, yrg):

    print "Start drawing"
    plt.figure(figsize=(30, 30), dpi=100, facecolor='w', edgecolor='k')
    hist, xx, yy = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=2000, range=[xrg, yrg])

    plt.imshow(np.log(hist))
    plt.savefig(data.DATA_PATH + "/analysis/xyhmap2.png")


if __name__ == "__main__":
    extract_coordinates(n_coordinates=32502730)

    coordinates = cPickle.load(open(data.path + "/coordinates_array.pkl", "rb"))
    xrg = [-8.75, -8.55]
    yrg = [41.05, 41.25]
    draw_map(coordinates, xrg, yrg)
