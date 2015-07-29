#!/usr/bin/env python

import csv
import os

from fuel.iterator import DataIterator
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream

from data.hdf5 import TaxiDataset
import data

dest_outfile = open(os.path.join(data.path, 'test_answer.csv'), 'w')
dest_outcsv = csv.writer(dest_outfile)
dest_outcsv.writerow(["TRIP_ID", "LATITUDE", "LONGITUDE"])

dataset = TaxiDataset('test', 'tvt.hdf5',
                     sources=('trip_id', 'longitude', 'latitude',
                              'destination_longitude', 'destination_latitude'))
it = DataIterator(DataStream(dataset), iter(xrange(dataset.num_examples)), as_dict=True)

for v in it:
    # print v
    dest_outcsv.writerow([v['trip_id'], v['destination_latitude'],
                                        v['destination_longitude']])

dest_outfile.close()

