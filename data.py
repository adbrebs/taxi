import ast, csv
import socket
import fuel
from enum import Enum
from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.iterator import DataIterator

if socket.gethostname() == "adeb.laptop":
    DATA_PATH = "/Users/adeb/data/taxi"
else:
    DATA_PATH="/data/lisatmp3/auvolat/taxikaggle"

client_ids = {int(x): y+1 for y, x in enumerate(open(DATA_PATH+"/client_ids.txt"))}

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
        for idx, (_, constructor) in enumerate(self.columns):
            values.append(constructor(line[idx]))
        return tuple(values)

taxi_columns = [
    ("trip_id", lambda x: x),
    ("call_type", CallType.from_data),
    ("origin_call", lambda x: 0 if x == '' or x == 'NA' else client_ids[int(x)]),
    ("origin_stand", lambda x: 0 if x == '' or x == 'NA' else int(x)),
    ("taxi_id", int),
    ("timestamp", int),
    ("day_type", DayType.from_data),
    ("missing_data", lambda x: x[0] == 'T'),
    ("polyline", lambda x: map(tuple, ast.literal_eval(x))),
]

train_files=["%s/split/train-%02d.csv" % (DATA_PATH, i) for i in range(100)]
valid_files=["%s/split/valid.csv" % (DATA_PATH,)]
train_data=TaxiData(train_files, taxi_columns)
valid_data=TaxiData(valid_files, taxi_columns)

def train_it():
    return DataIterator(DataStream(train_data))

def test_it():
    return DataIterator(DataStream(valid_data))
