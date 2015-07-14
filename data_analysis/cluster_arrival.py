#!/usr/bin/env python2
import numpy
import cPickle
import scipy.misc
import os

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle

import data
from data.hdf5 import taxi_it
from data.transformers import add_destination

print "Generating arrival point list"
dests = []
for v in taxi_it("train"):
    if len(v['latitude']) == 0: continue
    dests.append([v['latitude'][-1], v['longitude'][-1]])
pts = numpy.array(dests)

with open(os.path.join(data.path, "arrivals.pkl"), "w") as f:
    cPickle.dump(pts, f, protocol=cPickle.HIGHEST_PROTOCOL)

print "Doing clustering"
bw = estimate_bandwidth(pts, quantile=.1, n_samples=1000)
print bw
bw = 0.001 # (

ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(pts)
cluster_centers = ms.cluster_centers_

print "Clusters shape: ", cluster_centers.shape

with open(os.path.join(data.path, "arrival-clusters.pkl"), "w") as f:
    cPickle.dump(cluster_centers, f, protocol=cPickle.HIGHEST_PROTOCOL)

