import cPickle
import numpy as np
import matplotlib.pyplot as plt

import data
from data.hdf5 import taxi_it


def compute_number_coordinates():

    # Count the number of coordinates
    n_coordinates = 0
    for ride in taxi_it('train'):
        n_coordinates += len(ride['latitude'])
    print n_coordinates

    return n_coordinates


def extract_coordinates(n_coordinates=None):
    """Extract coordinates from the dataset and store them in a numpy array"""

    if n_coordinates is None:
        n_coordinates = compute_number_coordinates()

    coordinates = np.zeros((n_coordinates, 2), dtype="float32")

    c = 0
    for ride in taxi_it('train'):
        for point in zip(ride['latitude'], ride['longitude']):
            coordinates[c] = point
            c += 1

    print c

    cPickle.dump(coordinates, open(data.path + "/coordinates_array.pkl", "wb"))


def draw_map(coordinates, xrg, yrg):

    print "Start drawing"
    plt.figure(figsize=(30, 30), dpi=100, facecolor='w', edgecolor='k')
    hist, xx, yy = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=2000, range=[xrg, yrg])

    plt.imshow(np.log(hist))
    plt.gca().invert_yaxis()
    plt.savefig(data.path + "/analysis/xyhmap2.png")


if __name__ == "__main__":
    extract_coordinates(n_coordinates=83409386)

    coordinates = cPickle.load(open(data.path + "/coordinates_array.pkl", "rb"))
    xrg = [41.05, 41.25]
    yrg = [-8.75, -8.55]
    draw_map(coordinates, xrg, yrg)
