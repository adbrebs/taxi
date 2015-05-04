import matplotlib.pyplot as plt
import numpy
import cPickle
import scipy.misc

print "Loading data..."
with open("train.pkl") as f: normal = cPickle.load(f)

print "Extracting x and y"
# xes = [c[0] for l in normal for c in l[-1]]
# yes = [c[1] for l in normal for c in l[-1]]
xes = [l[-1][-1][0] for l in normal if len(l[-1]) > 0]
yes = [l[-1][-1][1] for l in normal if len(l[-1]) > 0]

xrg = [-8.80, -8.50]
yrg = [41.00, 41.30]

#print "Doing 1d x histogram"
#plt.clf(); plt.hist(xes, bins=2000, range=xrg); plt.savefig("xhist_dest.pdf")
#print "Doing 1d y histogram"
#plt.clf(); plt.hist(yes, bins=2000, range=yrg); plt.savefig("yhist_dest.pdf")

print "Doing 2d histogram"
hist, xx, yy = numpy.histogram2d(xes, yes, bins=4000, range=[xrg, yrg])

# import ipdb; ipdb.set_trace()

print "Imshow"
plt.clf(); plt.imshow(numpy.log(hist)); plt.savefig("xyhmap_dest_x.png", dpi=600)

print "Imsave"
scipy.misc.imsave("xymap_dest_2_x.png", numpy.log(hist + 1))
