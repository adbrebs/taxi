import ast
import csv
import numpy
import os

from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.iterator import DataIterator

import data
from data.hdf5 import origin_call_normalize, taxi_id_normalize


class TaxiData(Dataset):
    example_iteration_scheme=None

    class State:
        __slots__ = ('file', 'index', 'reader')

    def __init__(self, pathes, columns, has_header=False):
        if not isinstance(pathes, list):
            pathes=[pathes]
        assert len(pathes)>0
        self.columns=columns
        self.provides_sources = tuple(map(lambda x: x[0], columns))
        self.pathes=pathes
        self.has_header=has_header
        super(TaxiData, self).__init__()

    def open(self):
        state=self.State()
        state.file=open(self.pathes[0])
        state.index=0
        state.reader=csv.reader(state.file)
        if self.has_header:
            state.reader.next()
        return state

    def close(self, state):
        state.file.close()

    def reset(self, state):
        if state.index==0:
            state.file.seek(0)
        else:
            state.index=0
            state.file.close()
            state.file=open(self.pathes[0])
        state.reader=csv.reader(state.file)
        return state

    def get_data(self, state, request=None):
        if request is not None:
            raise ValueError
        try:
            line=state.reader.next()
        except (ValueError, StopIteration):
            # print state.index
            state.file.close()
            state.index+=1
            if state.index>=len(self.pathes):
                raise StopIteration
            state.file=open(self.pathes[state.index])
            state.reader=csv.reader(state.file)
            if self.has_header:
                state.reader.next()
            return self.get_data(state)

        values = []
        for _, constructor in self.columns:
            values.append(constructor(line))
        return tuple(values)

taxi_columns = [
    ("trip_id", lambda l: l[0]),
    ("call_type", lambda l: ord(l[1])-ord('A')),
    ("origin_call", lambda l: 0 if l[2] == '' or l[2] == 'NA' else origin_call_normalize(int(l[2]))),
    ("origin_stand", lambda l: 0 if l[3] == '' or l[3] == 'NA' else int(l[3])),
    ("taxi_id", lambda l: taxi_id_normalize(int(l[4]))),
    ("timestamp", lambda l: int(l[5])),
    ("day_type", lambda l: ord(l[6])-ord('A')),
    ("missing_data", lambda l: l[7][0] == 'T'),
    ("polyline", lambda l: map(tuple, ast.literal_eval(l[8]))),
    ("longitude", lambda l: map(lambda p: p[0], ast.literal_eval(l[8]))),
    ("latitude", lambda l: map(lambda p: p[1], ast.literal_eval(l[8]))),
]

taxi_columns_valid = taxi_columns + [
    ("destination_longitude", lambda l: numpy.float32(float(l[9]))),
    ("destination_latitude", lambda l: numpy.float32(float(l[10]))),
    ("time", lambda l: int(l[11])),
]

train_file = os.path.join(data.path, 'train.csv')
valid_file = os.path.join(data.path, 'valid2-cut.csv')
test_file = os.path.join(data.path, 'test.csv')

train_data=TaxiData(train_file, taxi_columns, has_header=True)
valid_data = TaxiData(valid_file, taxi_columns_valid)
test_data = TaxiData(test_file, taxi_columns, has_header=True)

with open(os.path.join(data.path, 'valid2-cut-ids.txt')) as f:
    valid_trips = [l for l in f]

def train_it():
    return DataIterator(DataStream(train_data))

def test_it():
    return DataIterator(DataStream(valid_data))
