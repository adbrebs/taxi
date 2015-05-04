import matplotlib.pyplot as plt
import numpy
import cPickle
import scipy.misc

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle

print "Reading arrival point list"
with open("arrivals.pkl") as f:
    pts = cPickle.load(f)

print "Doing clustering"
bw = estimate_bandwidth(pts, quantile=.1, n_samples=1000)
print bw
bw = 0.001

ms = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=5)
ms.fit(pts)
cluster_centers = ms.cluster_centers_

print "Clusters shape: ", cluster_centers.shape

with open("arrival-cluters.pkl", "w") as f:
    cPickle.dump(cluster_centers, f, protocol=cPickle.HIGHEST_PROTOCOL)

