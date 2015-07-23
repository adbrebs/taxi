#!/usr/bin/env python2
# Separate the training set into a Training Valid and Test set

import os
import sys
import importlib
import cPickle

import h5py
import numpy
import theano

import data
from data.hdf5 import TaxiDataset
from error import hdist


native_fields = {
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
}

all_fields = {
    'path_len': numpy.int16,
    'cluster': numpy.int16,
    'destination_latitude': numpy.float32,
    'destination_longitude': numpy.float32,
    'travel_time': numpy.int32,
}

all_fields.update(native_fields)

def cut_me_baby(train, cuts, excl={}):
    dset = {}
    cuts.sort()
    cut_id = 0
    for i in xrange(data.train_size):
        if i%10000==0 and i!=0:
            print >> sys.stderr, 'cut: {:d} done'.format(i)
        if i in excl:
            continue
        time = train['timestamp'][i]
        latitude = train['latitude'][i]
        longitude = train['longitude'][i]

        if len(latitude) == 0:
            continue

        end_time = time + 15 * (len(latitude) - 1)

        while cuts[cut_id] < time:
            if cut_id >= len(cuts)-1:
                return dset
            cut_id += 1

        if end_time < cuts[cut_id]:
            continue
        else:
            dset[i] = (cuts[cut_id] - time) / 15 + 1

    return dset

def make_tvt(test_cuts_name, valid_cuts_name, outpath):
    trainset = TaxiDataset('train')
    traindata = trainset.get_data(None, slice(0, trainset.num_examples))
    idsort = traindata[trainset.sources.index('timestamp')].argsort()

    traindata = dict(zip(trainset.sources, (t[idsort] for t in traindata)))

    print >> sys.stderr, 'test cut begin'
    test_cuts = importlib.import_module('.%s' % test_cuts_name, 'data.cuts').cuts
    test = cut_me_baby(traindata, test_cuts)

    print >> sys.stderr, 'valid cut begin'
    valid_cuts = importlib.import_module('.%s' % valid_cuts_name, 'data.cuts').cuts
    valid = cut_me_baby(traindata, valid_cuts, test)

    test_size = len(test)
    valid_size = len(valid)
    train_size = data.train_size - test_size - valid_size

    print ' set   | size    | ratio'
    print ' ----- | ------- | -----'
    print ' train | {:>7d} | {:>5.3f}'.format(train_size, float(train_size)/data.train_size)
    print ' valid | {:>7d} | {:>5.3f}'.format(valid_size, float(valid_size)/data.train_size)
    print ' test  | {:>7d} | {:>5.3f}'.format(test_size , float(test_size )/data.train_size)

    with open(os.path.join(data.path, 'arrival-clusters.pkl'), 'r') as f:
        clusters = cPickle.load(f)

    print >> sys.stderr, 'compiling cluster assignment function'
    latitude = theano.tensor.scalar('latitude')
    longitude = theano.tensor.scalar('longitude')
    coords = theano.tensor.stack(latitude, longitude).dimshuffle('x', 0)
    parent = theano.tensor.argmin(hdist(clusters, coords))
    cluster = theano.function([latitude, longitude], parent)

    train_clients = set()

    print >> sys.stderr, 'preparing hdf5 data'
    hdata = {k: numpy.empty(shape=(data.train_size,), dtype=v) for k, v in all_fields.iteritems()}

    train_i = 0
    valid_i = train_size
    test_i = train_size + valid_size

    print >> sys.stderr, 'write: begin'
    for idtraj in xrange(data.train_size):
        if idtraj%10000==0 and idtraj!=0:
            print >> sys.stderr, 'write: {:d} done'.format(idtraj)
        in_test = idtraj in test
        in_valid = not in_test and idtraj in valid
        in_train = not in_test and not in_valid

        if idtraj in test:
            i = test_i
            test_i += 1
        elif idtraj in valid:
            i = valid_i
            valid_i += 1
        else:
            train_clients.add(traindata['origin_call'][idtraj])
            i = train_i
            train_i += 1

        trajlen = len(traindata['latitude'][idtraj])
        if trajlen == 0:
            hdata['destination_latitude'][i] = data.train_gps_mean[0]
            hdata['destination_longitude'][i] = data.train_gps_mean[1]
        else:
            hdata['destination_latitude'][i] = traindata['latitude'][idtraj][-1]
            hdata['destination_longitude'][i] = traindata['longitude'][idtraj][-1]
        hdata['travel_time'][i] = trajlen

        for field in native_fields:
            val = traindata[field][idtraj]
            if field in ['latitude', 'longitude']:
                if in_test:
                    val = val[:test[idtraj]]
                elif in_valid:
                    val = val[:valid[idtraj]]
            hdata[field][i] = val

        plen = len(hdata['latitude'][i])
        hdata['path_len'][i] = plen
        hdata['cluster'][i] = -1 if plen==0 else cluster(hdata['latitude'][i][0], hdata['longitude'][i][0])

    print >> sys.stderr, 'write: end'

    print >> sys.stderr, 'removing useless origin_call'
    for i in xrange(train_size, data.train_size):
        if hdata['origin_call'][i] not in train_clients:
            hdata['origin_call'][i] = 0

    print >> sys.stderr, 'preparing split array'

    split_array = numpy.empty(len(all_fields)*3, dtype=numpy.dtype([
        ('split', 'a', 64),
        ('source', 'a', 21),
        ('start', numpy.int64, 1),
        ('stop', numpy.int64, 1),
        ('indices', h5py.special_dtype(ref=h5py.Reference)),
        ('available', numpy.bool, 1),
        ('comment', 'a', 1)]))

    flen = len(all_fields)
    for i, field in enumerate(all_fields):
        split_array[i]['split'] = 'train'.encode('utf8')
        split_array[i+flen]['split'] = 'valid'.encode('utf8')
        split_array[i+2*flen]['split'] = 'test'.encode('utf8')
        split_array[i]['start'] = 0
        split_array[i]['stop'] = train_size
        split_array[i+flen]['start'] = train_size
        split_array[i+flen]['stop'] = train_size + valid_size
        split_array[i+2*flen]['start'] = train_size + valid_size
        split_array[i+2*flen]['stop'] = train_size + valid_size + test_size

        for d in [0, flen, 2*flen]:
            split_array[i+d]['source'] = field.encode('utf8')

    split_array[:]['indices'] = None
    split_array[:]['available'] = True
    split_array[:]['comment'] = '.'.encode('utf8')

    print >> sys.stderr, 'writing hdf5 file'
    file = h5py.File(outpath, 'w')
    for k in all_fields.keys():
        file.create_dataset(k, data=hdata[k], maxshape=(data.train_size,))

    file.attrs['split'] = split_array

    file.flush()
    file.close()

if __name__ == '__main__':
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print >> sys.stderr, 'Usage: %s test_cutfile valid_cutfile [outfile]' % sys.argv[0]
        sys.exit(1)
    outpath = os.path.join(data.path, 'tvt.hdf5') if len(sys.argv) < 4 else sys.argv[3]
    make_tvt(sys.argv[1], sys.argv[2], outpath)
