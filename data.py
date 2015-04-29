import ast, csv
import socket
import fuel
import numpy
import h5py
from enum import Enum
from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.iterator import DataIterator
import theano

if socket.gethostname() == "adeb.laptop":
    DATA_PATH = "/Users/adeb/data/taxi"
else:
    DATA_PATH="/data/lisatmp3/auvolat/taxikaggle"

H5DATA_PATH = '/data/lisatmp3/simonet/taxi/data.hdf5'

porto_center = numpy.array([41.1573, -8.61612], dtype=theano.config.floatX)
data_std = numpy.sqrt(numpy.array([0.00549598, 0.00333233], dtype=theano.config.floatX))

n_clients = 57124
n_train_clients = 57105
n_stands = 63

dataset_size = 1710670

# ---- Read client IDs and create reverse dictionnary

def make_client_ids():
    f = h5py.File(H5DATA_PATH, "r")
    l = f['unique_origin_call']
    r = {}
    for i in range(l.shape[0]):
        r[l[i]] = i
    return r

client_ids = make_client_ids()

def get_client_id(n):
    if n in client_ids and client_ids[n] <= n_train_clients:
        return client_ids[n]
    else:
        return 0

class CallType(Enum):
    CENTRAL = 0
    STAND = 1
    STREET = 2

    @classmethod
    def from_data(cls, val):
        if val=='A':
            return cls.CENTRAL
        elif val=='B':
            return cls.STAND
        elif val=='C':
            return cls.STREET

    @classmethod
    def to_data(cls, val):
        if val==cls.CENTRAL:
            return 'A'
        elif val==cls.STAND:
            return 'B'
        elif val==cls.STREET:
            return 'C'

class DayType(Enum):
    NORMAL = 0
    HOLIDAY = 1
    HOLIDAY_EVE = 2

    @classmethod
    def from_data(cls, val):
        if val=='A':
            return cls.NORMAL
        elif val=='B':
            return cls.HOLIDAY
        elif val=='C':
            return cls.HOLIDAY_EVE

    @classmethod
    def to_data(cls, val):
        if val==cls.NORMAL:
            return 'A'
        elif val==cls.HOLIDAY:
            return 'B'
        elif val==cls.HOLIDAY_EVE:
            return 'C'

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
    ("call_type", lambda l: CallType.from_data(l[1])),
    ("origin_call", lambda l: 0 if l[2] == '' or l[2] == 'NA' else get_client_id(int(l[2]))),
    ("origin_stand", lambda l: 0 if l[3] == '' or l[3] == 'NA' else int(l[3])),
    ("taxi_id", lambda l: int(l[4])),
    ("timestamp", lambda l: int(l[5])),
    ("day_type", lambda l: DayType.from_data(l[6])),
    ("missing_data", lambda l: l[7][0] == 'T'),
    ("polyline", lambda l: map(tuple, ast.literal_eval(l[8]))),
    ("longitude", lambda l: map(lambda p: p[0], ast.literal_eval(l[8]))),
    ("latitude", lambda l: map(lambda p: p[1], ast.literal_eval(l[8]))),
]

taxi_columns_valid = taxi_columns + [
    ("destination_longitude", lambda l: float(l[9])),
    ("destination_latitude", lambda l: float(l[10])),
    ("time", lambda l: int(l[11])),
]

train_files=["%s/split/train-%02d.csv" % (DATA_PATH, i) for i in range(100)]
valid_files=["%s/split/valid.csv" % (DATA_PATH,)]
test_file="%s/test.csv" % (DATA_PATH,)

train_data=TaxiData(train_files, taxi_columns)
valid_data = TaxiData(valid_files, taxi_columns_valid)
test_data = TaxiData(test_file, taxi_columns, has_header=True)

def train_it():
    return DataIterator(DataStream(train_data))

def test_it():
    return DataIterator(DataStream(valid_data))
