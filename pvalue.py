#!/usr/bin/env python

import os
import sys

import math
import numpy

import data

# Haversine distance calculation
# --------- -------- -----------

rearth = 6371.
deg2rad = 3.141592653589793 / 180.

def hdist(a, b):
    lat1 = a[:, 0] * deg2rad
    lon1 = a[:, 1] * deg2rad
    lat2 = b[:, 0] * deg2rad
    lon2 = b[:, 1] * deg2rad

    dlat = abs(lat1-lat2)
    dlon = abs(lon1-lon2)

    al = numpy.sin(dlat/2)**2  + numpy.cos(lat1) * numpy.cos(lat2) * (numpy.sin(dlon/2)**2)
    d = numpy.arctan2(numpy.sqrt(al), numpy.sqrt(1.-al))

    hd = 2. * rearth * d

    return hd


# Read the inputs
# ---- --- ------

def readcsv(f):
    return numpy.genfromtxt(f, delimiter=',', skip_header=1)[:, 1:3]

answer = readcsv(os.path.join(data.path, 'test_answer.csv'))

tables = [readcsv(f) for f in sys.argv if '.csv' in f]
etables = [hdist(t, answer) for t in tables]

# Calculate p-values
# --------- --------

pvalue = numpy.zeros((len(tables), len(tables)))

for i, a in enumerate(etables):
    for j, b in enumerate(etables):
        if i == j:
            continue
        d = b - a
        var = (numpy.mean((a - numpy.mean(a))**2)
                + numpy.mean((b - numpy.mean(b))**2)) / 2.
        pv = 1 - .5 * (1 + math.erf(numpy.mean(d) / numpy.sqrt(2 * var)))
        pvalue[i, j] = pv

print pvalue
