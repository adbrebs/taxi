import os
import json
import getpass
from datetime import datetime
import itertools

import numpy

import data


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if type(o).__module__ == numpy.__name__:
            return o.item()
        super(NumpyEncoder, self).default(o)


class EGJ(object):
    def save(self, path=getpass.getuser(), append=False):
        path = os.path.join(data.path, 'visualizer', path)
        if append:
            if not os.path.isdir(path):
                raise ValueError("Can't append to the given directory")
            name = str(1+max(map(int, filter(str.isdigit, os.listdir(path)))+[-1]))
            path = os.path.join(path, name)
        else:
            while os.path.isdir(path):
                path = os.path.join(path, '0')

        with open(path, 'w') as f:
            self.write(f)

    def write(self, file):
        file.write(json.dumps(self.object(), cls=NumpyEncoder))

    def type(self):
        return 'raw'

    def options(self):
        return []

    def object(self):
        return {
                'type': self.type(),
                'data': {
                    'type': 'FeatureCollection',
                    'crs': {
                        'type': 'name',
                        'properties': {
                            'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'
                        }
                    },
                    'features': self.features()
                }
            }


class Point(EGJ):
    def __init__(self, latitude, longitude, info=None):
        self.latitude = latitude
        self.longitude = longitude
        self.info = info

    def features(self):
        d = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [self.longitude, self.latitude]
                }
            }
        if self.info is not None:
            d['properties'] = { 'info': self.info }
        return [d]


class Path(EGJ):
    def __init__(self, path, info=''):
        self.path = path
        self.info = info

    def features(self):
        info = self.info + '''trip_id: %(trip_id)s<br>
            call_type: %(call_type_f)s<br>
            origin_call: %(origin_call)d<br>
            origin_stand: %(origin_stand)d<br>
            taxi_id: %(taxi_id)d<br>
            timestamp: %(timestamp_f)s<br>
            day_type: %(day_type_f)s<br>
            missing_data: %(missing_data)d<br>''' \
            % dict(self.path,
                call_type_f = ['central', 'stand', 'street'][self.path['call_type']],
                timestamp_f = datetime.fromtimestamp(self.path['timestamp']).strftime('%c'),
                day_type_f = ['normal', 'holiday', 'holiday eve'][self.path['day_type']])

        return [{
                'type': 'Feature',
                'properties': {
                    'info': info,
                    'display': 'path',
                    'timestamp': self.path['timestamp']
                },
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[lon, lat] for (lat, lon) in zip(self.path['latitude'], self.path['longitude'])]
                }
            }]


class Vlist(EGJ, list):
    def __init__(self, cluster=False, heatmap=False, distrib=False, *args):
        list.__init__(self, *args)
        self.cluster = cluster
        self.heatmap = heatmap
        self.distrib = distrib

    def type(self):
        ts = self.cluster + self.heatmap + self.distrib 
        assert ts <= 1
        if ts > 0:
            if all(isinstance(c, Point) for c in self):
                if self.cluster:
                    return 'cluster'
                elif self.heatmap:
                    return 'heatmap'
                elif self.distrib:
                    return 'point distribution'
            else:
                raise ValueError('Building a %s with something that is not a Point' % ('cluster' if self.cluster else 'heatmap'))
        else:
            return 'raw'

    def features(self):
        return list(itertools.chain.from_iterable(p.features() for p in self))
