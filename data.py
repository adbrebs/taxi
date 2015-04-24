import ast, csv
import fuel
from enum import Enum
from fuel.datasets import Dataset
from fuel.streams import DataStream
from fuel.iterator import DataIterator

PREFIX="/data/lisatmp3/auvolat/taxikaggle"

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

    def __init__(self, path):
        self.path=path
        super(TaxiData, self).__init__()

    def open(self):
        file=open(self.path)
        reader=csv.reader(file)
        reader.next() # Skip header
        return (file, reader)

    def close(self, state):
        state[0].close()

    def reset(self, state):
        state[0].seek(0)
        state[1]=csv.reader(state[0])
        return state

    def get_data(self, state, request=None):
        if request is not None:
            raise ValueError
        line=state[1].next()
        line[1]=CallType.from_data(line[1]) # call_type
        line[2]=0 if line[2]=='' or line[2]=='NA' else int(line[2]) # origin_call
        line[3]=0 if line[3]=='' or line[3]=='NA' else int(line[3]) # origin_stand
        line[4]=int(line[4]) # taxi_id
        line[5]=int(line[5]) # timestamp
        line[6]=DayType.from_data(line[6]) # day_type
        line[7]=line[7][0]=='T' # missing_data
        line[8]=map(tuple, ast.literal_eval(line[8])) # polyline
        return tuple(line)

train_data=TaxiData(PREFIX+'/train.csv')
test_data=TaxiData(PREFIX+'/test.csv')

def train_it():
    return DataIterator(DataStream(train_data))

def test_it():
    return DataIterator(DataStream(test_data))
