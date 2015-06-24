#!/usr/bin/env python

import cPickle
import sys
import os
import importlib
import csv

from blocks.model import Model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print >> sys.stderr, 'Usage: %s config' % sys.argv[0]
        sys.exit(1)
    model_name = sys.argv[1]
    config = importlib.import_module('.%s' % model_name, 'config')
    model_config = config.Model(config)

    stream = config.Stream(config)
    inputs = stream.inputs()
    outputs = model_config.predict.outputs
    req_vars_test = model_config.predict.inputs + ['trip_id']
    test_stream = stream.test(req_vars_test)

    model = Model(model_config.predict(**inputs))
    with open(os.path.join('model_data', "{}.pkl".format(model_name))) as f:
        parameters = cPickle.load(f)
    model.set_param_values(parameters)

    if 'destination' in outputs:
        dest_outfile = open(os.path.join('output', 'test-dest-output-%s.csv' % model_name), 'w')
        dest_outcsv = csv.writer(dest_outfile)
        dest_outcsv.writerow(["TRIP_ID", "LATITUDE", "LONGITUDE"])
    if 'duration' in outputs:
        time_outfile = open(os.path.join('output', 'test-time-output-%s.csv' % model_name), 'w')
        time_outcsv = csv.writer(time_outfile)
        time_outcsv.writerow(["TRIP_ID", "TRAVEL_TIME"])

    function = model.get_theano_function()
    for d in test_stream.get_epoch_iterator(as_dict=True):
        input_values = [d[k.name] for k in model.inputs]
        output_values = function(*input_values)
        if 'destination' in outputs:
            destination = output_values[outputs.index('destination')]
            dest_outcsv.writerow([d['trip_id'][0], destination[0, 0], destination[0, 1]])
        if 'duration' in outputs:
            duration = output_values[outputs.index('duration')]
            time_outcsv.writerow([d['trip_id'][0], int(round(duration[0]))])

    if 'destination' in outputs:
        dest_outfile.close()
    if 'duration' in outputs:
        time_outfile.close()
