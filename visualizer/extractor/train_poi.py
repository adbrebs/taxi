#!/usr/bin/env python

import os

import data
from data.hdf5 import TaxiDataset
from visualizer import Path


poi = {
    'longest': 1492417
}

if __name__ == '__main__':
    prefix = os.path.join(data.path, 'visualizer', 'Train POI')
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    d = TaxiDataset('train')
    for (k, v) in poi.items():
        Path(d.extract(v)).save(os.path.join('Train POI', k))
