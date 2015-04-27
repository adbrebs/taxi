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
    provides_sources= ("trip_id","call_type","origin_call","origin_stand","taxi_id","timestamp","day_type","missing_data","polyline")
    example_iteration_scheme=None

    class State:
        __slots__ = ('file', 'index', 'reader')

    def __init__(self, pathes, has_header=False):
        if not isinstance(pathes, list):
            pathes=[pathes]
        assert len(pathes)>0
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

        line[1]=CallType.from_data(line[1]) # call_type
        line[2]=0 if line[2]=='' or line[2]=='NA' else client_ids[int(line[2])] # origin_call
        line[3]=0 if line[3]=='' or line[3]=='NA' else int(line[3]) # origin_stand
        line[4]=int(line[4]) # taxi_id
        line[5]=int(line[5]) # timestamp
        line[6]=DayType.from_data(line[6]) # day_type
        line[7]=line[7][0]=='T' # missing_data
        line[8]=map(tuple, ast.literal_eval(line[8])) # polyline
        return tuple(line)

train_files=["%s/split/train-%02d.csv" % (DATA_PATH, i) for i in range(100)]
valid_files=["%s/split/valid.csv" % (DATA_PATH,)]
train_data=TaxiData(train_files)
valid_data=TaxiData(valid_files)

def train_it():
    return DataIterator(DataStream(train_data))

def test_it():
    return DataIterator(DataStream(valid_data))
