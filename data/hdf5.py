import os

import h5py
from fuel.datasets import H5PYDataset
from fuel.iterator import DataIterator
from fuel.schemes import SequentialExampleScheme
from fuel.streams import DataStream

import data


class TaxiDataset(H5PYDataset):
    def __init__(self, which_set, filename='data.hdf5', **kwargs):
        self.filename = filename
        kwargs.setdefault('load_in_memory', True)
        super(TaxiDataset, self).__init__(self.data_path, (which_set,), **kwargs)

    @property
    def data_path(self):
        return os.path.join(data.path, self.filename)

    def extract(self, request):
        if not self.load_in_memory:
            raise ValueError('extract called on a dataset not loaded in memory')
        return dict(zip(self.sources, self.get_data(None, request)))

class TaxiStream(DataStream):
    def __init__(self, which_set, filename='data.hdf5', iteration_scheme=None, **kwargs):
        dataset = TaxiDataset(which_set, filename, **kwargs)
        if iteration_scheme is None:
            iteration_scheme = SequentialExampleScheme(dataset.num_examples)
        super(TaxiStream, self).__init__(dataset, iteration_scheme=iteration_scheme)

_origin_calls = None
_reverse_origin_calls = None

def origin_call_unnormalize(x):
    if _origin_calls is None:
        _origin_calls = h5py.File(os.path.join(data.path, 'data.hdf5'), 'r')['unique_origin_call']
    return _origin_calls[x]

def origin_call_normalize(x):
    if _reverse_origin_calls is None:
        origin_call_unnormalize(0)
        _reverse_origin_calls = { _origin_calls[i]: i for i in range(_origin_calls.shape[0]) }
    return _reverse_origin_calls[x]

_taxi_ids = None
_reverse_taxi_ids = None

def taxi_id_unnormalize(x):
    if _taxi_ids is None:
        _taxi_ids = h5py.File(os.path.join(data.path, 'data.hdf5'), 'r')['unique_taxi_id']
    return _taxi_ids[x]

def taxi_id_normalize(x):
    if _reverse_taxi_ids is None:
        taxi_id_unnormalize(0)
        _reverse_taxi_ids = { _taxi_ids[i]: i for i in range(_taxi_ids.shape[0]) }
    return _reverse_taxi_ids[x]

def taxi_it(which_set, filename='data.hdf5', sub=None, as_dict=True):
    dataset = TaxiDataset(which_set, filename)
    if sub is None:
        sub = xrange(dataset.num_examples)
    return DataIterator(DataStream(dataset), iter(sub), as_dict)
