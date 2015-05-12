#!/usr/bin/env python

from data.hdf5 import taxi_it
from visualizer import Vlist, Point


if __name__ == '__main__':
    it = taxi_it('stands')
    next(it) # Ignore the "no stand" entry

    points = Vlist()
    for (i, line) in enumerate(it):
        points.append(Point(line['stands_latitude'], line['stands_longitude'], 'Stand (%d): %s' % (i+1, line['stands_name'])))
    points.save('stands')
