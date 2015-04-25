import matplotlib.pyplot as plt
import numpy
import cPickle
import scipy

print "Loading data..."
with open("../train_normal.pkl") as f: normal = cPickle.load(f)

print "Extracting x and y"
xes = [c[0] for l in normal for c in l[-1]]
yes = [c[1] for l in normal for c in l[-1]]

xrg = [-8.75, -8.55]
yrg = [41.05, 41.25]

print "Doing 1d histogram"
#plt.clf(); plt.hist(xes, bins=1000, range=xrg); plt.savefig("xhist.pdf")
#plt.clf(); plt.hist(yes, bins=1000, range=yrg); plt.savefig("yhist.pdf")

print "Doing 2d histogram"
#plt.clf(); plt.hist2d(xes, yes, bins=500, range=[xrg, yrg]); plt.savefig("xymap.pdf")

hist, xx, yy = numpy.histogram2d(xes, yes, bins=2000, range=[xrg, yrg])

import ipdb; ipdb.set_trace()

plt.clf(); plt.imshow(numpy.log(hist)); plt.savefig("xyhmap.pdf")

scipy.misc.imsave("xymap.png", numpy.log(hist))
