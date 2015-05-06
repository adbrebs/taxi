import os

import h5py
import numpy
import theano


path = os.environ.get('TAXI_PATH', '/data/lisatmp3/auvolat/taxikaggle')
Polyline = h5py.special_dtype(vlen=theano.config.floatX)


# `wc -l test.csv` - 1 # Minus 1 to ignore the header
test_size = 320 

# `wc -l train.csv` - 1
train_size = 1710670 

# `wc -l metaData_taxistandsID_name_GPSlocation.csv`
stands_size = 64 # include 0 ("no origin_stands")

# `cut -d, -f 5 train.csv test.csv | sort -u | wc -l` - 1
taxi_id_size = 448 

# `cut -d, -f 3 train.csv test.csv | sort -u | wc -l` - 2
origin_call_size = 57125 # include 0 ("no origin_call")

# As printed by csv_to_hdf5.py
origin_call_train_size = 57106

train_gps_mean = numpy.array([41.1573, -8.61612], dtype=theano.config.floatX)
train_gps_std = numpy.sqrt(numpy.array([0.00549598, 0.00333233], dtype=theano.config.floatX))
