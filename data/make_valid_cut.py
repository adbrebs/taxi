#!/usr/bin/env python2
# Make a valid dataset by cutting the training set at specified timestamps

import os
import sys
import importlib

import h5py
import numpy

import data
from data.hdf5 import taxi_it


_fields = ['trip_id', 'call_type', 'origin_call', 'origin_stand', 'taxi_id', 'timestamp', 'day_type', 'missing_data', 'latitude', 'longitude', 'destination_latitude', 'destination_longitude', 'travel_time']

def make_valid(cutfile, outpath):
    cuts = importlib.import_module('.%s' % cutfile, 'data.cuts').cuts

    print "Number of cuts:", len(cuts)

    valid = []

    for line in taxi_it('train'):
        time = line['timestamp']
        latitude = line['latitude']
        longitude = line['longitude']

        if len(latitude) == 0:
            continue

        for ts in cuts:
            if time <= ts and time + 15 * (len(latitude) - 1) >= ts:
                # keep it
                n = (ts - time) / 15 + 1
                line.update({
                    'latitude': latitude[:n],
                    'longitude': longitude[:n],
                    'destination_latitude': latitude[-1],
                    'destination_longitude': longitude[-1],
                    'travel_time': 15 * (len(latitude)-1)
                })
                valid.append(line)
                break

    print "Number of trips in validation set:", len(valid)
    
    file = h5py.File(outpath, 'a')
    clen = file['trip_id'].shape[0]
    alen = len(valid)
    for field in _fields:
        dset = file[field]
        dset.resize((clen + alen,))
        for i in xrange(alen):
            dset[clen + i] = valid[i][field]

    splits = file.attrs['split']
    slen = splits.shape[0]
    splits = numpy.resize(splits, (slen+len(_fields),))
    for (i, field) in enumerate(_fields):
        splits[slen+i]['split'] = ('cuts/%s' % cutfile).encode('utf8')
        splits[slen+i]['source'] = field.encode('utf8')
        splits[slen+i]['start'] = clen
        splits[slen+i]['stop'] = alen
        splits[slen+i]['indices'] = None
        splits[slen+i]['available'] = True
        splits[slen+i]['comment'] = '.'
    file.attrs['split'] = splits

    file.flush()
    file.close()

if __name__ == '__main__':
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print >> sys.stderr, 'Usage: %s cutfile [outfile]' % sys.argv[0]
        sys.exit(1)
    outpath = os.path.join(data.path, 'valid.hdf5') if len(sys.argv) < 3 else sys.argv[2]
    make_valid(sys.argv[1], outpath)
