#!/usr/bin/env python
# Make a valid dataset by cutting the training set at specified timestamps

import os
import sys
import importlib

import h5py
import numpy

import data
from data.hdf5 import taxi_it

import sqlite3

def make_valid(outpath):
    times = []
    for i, line in enumerate(taxi_it('train')):
        time = line['timestamp']
        latitude = line['latitude']

        if len(latitude) == 0:
            continue

        duration = 15 * (len(latitude) - 1)

        times.append((i, int(time), int(time + duration)))
        if i % 1000 == 0:
            print times[-1]


    with sqlite3.connect(outpath) as timedb:
        c = timedb.cursor()
        c.execute('''
                CREATE TABLE trip_times
                    (trip INTEGER, begin INTEGER, end INTEGER)
        ''')
        print "Adding data..."
        c.executemany('INSERT INTO trip_times(trip, begin, end) VALUES(?, ?, ?)', times)
        timedb.commit()
        print "Creating index..."
        c.execute('''CREATE INDEX trip_begin_index ON trip_times (begin)''')


if __name__ == '__main__':
    if len(sys.argv) < 1 or len(sys.argv) > 2:
        print >> sys.stderr, 'Usage: %s [outfile]' % sys.argv[0]
        sys.exit(1)
    outpath = os.path.join(data.path, 'time_index.db') if len(sys.argv) < 2 else sys.argv[1]
    make_valid(outpath)
