# Cuts the training dataset at the following timestamps :

cuts = [
    1376503200,
    1380616200,
    1381167900,
    1383364800,
    1387722600,
]

import random
import csv
import ast

f = open("train.csv")
fr = csv.reader(f)
_skip_header = fr.next()
g = open("cutvalid.csv", "w")
gw = csv.writer(g)

for l in fr:
    polyline = ast.literal_eval(l[-1])
    if len(polyline) == 0: continue
    time = int(l[5])
    for ts in cuts:
        if time <= ts and time + 15 * (len(polyline) - 1) >= ts:
            # keep it
            n = (ts - time) / 15 + 1
            cut = polyline[:n]
            row = l[:-1] + [
                        cut.__str__(),
                        polyline[-1][0],
                        polyline[-1][1],
                        15 * (len(polyline)-1)
                    ]
            print row
            gw.writerow(row)

f.close()
g.close()
