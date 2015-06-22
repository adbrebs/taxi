from fuel.schemes import IterationScheme
import sqlite3
import random
import os
from picklable_itertools import iter_

import data

first_time = 1372636853
last_time = 1404172787


class TaxiTimeCutScheme(IterationScheme):
    def __init__(self, num_cuts=100, dbfile=None, use_cuts=None):
        self.num_cuts = num_cuts
        self.dbfile = os.path.join(data.path, 'time_index.db') if dbfile == None else dbfile
        self.use_cuts = use_cuts

    def get_request_iterator(self):
        cuts = self.use_cuts
        if cuts == None:
            cuts = [random.randrange(first_time, last_time) for _ in range(self.num_cuts)]

        l = []
        with sqlite3.connect(self.dbfile) as db:
            c = db.cursor()
            for cut in cuts:
                part = [i for (i,) in
                    c.execute('SELECT trip FROM trip_times WHERE begin >= ? AND begin <= ? AND end >= ?',
                                (cut - 40000, cut, cut))]
                l = l + part
        random.shuffle(l)
                
        return iter_(l)

