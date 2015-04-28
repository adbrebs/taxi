# Takes valid-full.csv which is a subset of the lines of train.csv, formatted in the
# exact same way
# Outputs valid.csv which contains the polylines cut at an arbitrary location, and three
# new columns containing the destination point and the length in seconds of the original polyline
# (see contest definition for the time taken by a taxi along a polyline)

import random
import csv
import ast

with open("valid-full.csv") as f:
    vlines = [l for l in csv.reader(f)]

def make_valid_item(l):
    polyline = ast.literal_eval(l[-1])
    last = polyline[-1]
    cut_idx = random.randrange(len(polyline)-5)
    cut = polyline[:cut_idx+6]
    return l[:-1] + [
                        cut.__str__(),
                        last[0],
                        last[1],
                        15 * (len(polyline)-1),
                    ]

vlines = map(make_valid_item, filter(lambda l: (len(ast.literal_eval(l[-1])) > 5), vlines))

with open("valid.csv", "w") as f:
    wr = csv.writer(f)
    for r in vlines:
        wr.writerow(r)

with open("valid-solution.csv", "w") as f:
    wr = csv.writer(f)
    for r in vlines:
        wr.writerow([r[0], r[-2], r[-3]])
