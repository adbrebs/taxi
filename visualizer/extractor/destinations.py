#!/usr/bin/env python

from data.hdf5 import taxi_it
from visualizer import Vlist, Point


_sample_size = 5000

if __name__ == '__main__':
    points = Vlist(cluster=True)
    for line in taxi_it('train'):
        if len(line['latitude'])>0:
            points.append(Point(line['latitude'][-1], line['longitude'][-1]))
            if len(points) >= _sample_size:
                break
    points.save('destinations (cluster)')
    points.cluster = False
    points.heatmap = True
    points.save('destinations (heatmap)')
