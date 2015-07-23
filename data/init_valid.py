#!/usr/bin/env python2
# Initialize the valid hdf5

import os
import sys

import h5py
import numpy

import data


_fields = {
    'trip_id': 'S19',
    'call_type': numpy.int8,
    'origin_call': numpy.int32,
    'origin_stand': numpy.int8,
    'taxi_id': numpy.int16,
    'timestamp': numpy.int32,
    'day_type': numpy.int8,
    'missing_data': numpy.bool,
    'latitude': data.Polyline,
    'longitude': data.Polyline,
    'destination_latitude': numpy.float32,
    'destination_longitude': numpy.float32,
    'travel_time': numpy.int32,
}


def init_valid(path):
    h5file = h5py.File(path, 'w')
    
    for k, v in _fields.iteritems():
        h5file.create_dataset(k, (0,), dtype=v, maxshape=(None,))

    split_array = numpy.empty(len(_fields), dtype=numpy.dtype([
        ('split', 'a', 64),
        ('source', 'a', 21),
        ('start', numpy.int64, 1),
        ('stop', numpy.int64, 1),
        ('indices', h5py.special_dtype(ref=h5py.Reference)),
        ('available', numpy.bool, 1),
        ('comment', 'a', 1)]))

    split_array[:]['split'] = 'dummy'.encode('utf8')
    for (i, k) in enumerate(_fields.keys()):
        split_array[i]['source'] = k.encode('utf8')
    split_array[:]['start'] = 0
    split_array[:]['stop'] = 0
    split_array[:]['available'] = False
    split_array[:]['indices'] = None
    split_array[:]['comment'] = '.'.encode('utf8')
    h5file.attrs['split'] = split_array

    h5file.flush()
    h5file.close()

if __name__ == '__main__':
    if len(sys.argv) > 2:
        print >> sys.stderr, 'Usage: %s [file]' % sys.argv[0]
        sys.exit(1)
    init_valid(sys.argv[1] if len(sys.argv) == 2 else os.path.join(data.path, 'valid.hdf5'))
