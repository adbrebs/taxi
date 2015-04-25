import cPickle
import scipy
import numpy as np
import matplotlib.pyplot as plt

import data


def compute_number_coordinates():
    train_it = data.train_it()

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

    coordinates = np.zeros((n_coordinates, 2), dtype="float32")
    train_it = data.train_it()

    c = 0
    for ride in train_it:
        for point in ride[-1]:
            coordinates[c] = point
            c += 1

    cPickle.dump(coordinates, open(data.DATA_PATH + "/coordinates_array.pkl", "wb"))


def draw_map(coordinates, xrg, yrg):

    hist, xx, yy = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=2000, range=[xrg, yrg])

    plt.imshow(np.log(hist))
    plt.savefig(data.DATA_PATH + "/analysis/xyhmap.pdf")

    scipy.misc.imsave(data.DATA_PATH + "/analysis/xymap.png", np.log(hist))


if __name__ == "__main__":
    # extract_coordinates(n_coordinates=83360928)

    coordinates = cPickle.load(open(data.DATA_PATH + "/coordinates_array.pkl", "rb"))
    xrg = [-8.75, -8.55]
    yrg = [41.05, 41.25]
    draw_map(coordinates, xrg, yrg)
