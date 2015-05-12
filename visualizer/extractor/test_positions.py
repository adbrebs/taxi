#!/usr/bin/env python

from data.hdf5 import taxi_it
from visualizer import Vlist, Point


if __name__ == '__main__':
    points = Vlist(heatmap=True)
    for line in taxi_it('test'):
        for (lat, lon) in zip(line['latitude'], line['longitude']):
            points.append(Point(lat, lon))
    points.save('test positions')
