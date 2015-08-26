#!/usr/bin/env python

import os
import sys
import urllib
import SimpleHTTPServer
import SocketServer
from cStringIO import StringIO

import data
from data.hdf5 import TaxiDataset
from visualizer import Vlist, Path


visualizer_path = os.path.join(data.path, 'visualizer')
source_path = os.path.split(os.path.realpath(__file__))[0]

test_data = None
train_data = None

class VisualizerHTTPRequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def send_head(self):
        spath = self.path.split('?')[0]
        path = spath.split('/')[1:]
        if len(path) == 1:
            if path[0] == '':
                path[0] = 'index.html'
            file_path = os.path.join(source_path, path[0])
            return self.send_file(file_path)
        elif path[0] == 'ls':
            return self.send_datalist()
        elif path[0] == 'get':
            return self.send_file(os.path.join(visualizer_path, spath[5:]))
        elif path[0] == 'extract':
            return self.send_extract(spath[9:])

    def send_file(self, file_path):
        file_path = urllib.unquote(file_path)
        ctype = self.guess_type(file_path)

        try:
            f = open(file_path, 'rb')
        except IOError:
            self.send_error(404, 'File not found')
            return None
        try:
            self.send_response(200)
            self.send_header('Content-type', ctype)
            fs = os.fstat(f.fileno())
            self.send_header('Content-Length', str(fs[6]))
            self.send_header('Last-Modified', self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except:
            f.close()
            raise

    def send_datalist(self):
        l = []
        for path, subs, files in os.walk(visualizer_path):
            for file in files:
                mtime = os.stat('%s/%s' % (path, file))[8]
                l.append('{"path":["%s"],"name":"%s","mtime":%d}' % ('","'.join(path[len(visualizer_path):].split('/')), file, mtime))
        l.sort()
        f = StringIO()
        f.write("[")
        f.write(','.join(l))
        f.write("]")
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        encoding = sys.getfilesystemencoding()
        self.send_header("Content-type", "text/html; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

    def send_extract(self, query):
        f = StringIO()
        query = urllib.unquote(query)
        content = Vlist()
        for (i,sub) in enumerate(query.split(',')):
            r = sub.split('-')
            if len(r)==1:
                if sub.strip()[0].lower()=='t':
                    sub=sub.strip()[1:]
                    content.append(Path(test_data.extract(int(sub)), 'T%s<br>'%sub))
                else:
                    content.append(Path(train_data.extract(int(sub)), '%s<br>'%sub))
            elif len(r)==2:
                test = False
                if r[0].strip()[0].lower()=='t':
                    test = True
                    r[0]=r[0].strip()[1:]
                    if r[1].strip()[0].lower()=='t':
                        r[1]=r[1].strip()[1:]
                for i in xrange(int(r[0]), int(r[1])+1):
                    if test:
                        content.append(Path(test_data.extract(i), 'T%d<br>'%i))
                    else:
                        content.append(Path(train_data.extract(i), '%d<br>'%i))
            elif len(r)>2:
                self.send_error(404, 'File not found')
                return None
        content.write(f)
        length = f.tell()
        f.seek(0)
        self.send_response(200)
        encoding = sys.getfilesystemencoding()
        self.send_header("Content-type", "text/html; charset=%s" % encoding)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        return f

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >>sys.stderr, 'Usage: %s port [--no-hdf5]' % sys.argv[0]

    if '--no-hdf5' not in sys.argv:
        print >>sys.stderr, 'Loading dataset...',
        path = os.path.join(data.path, 'data.hdf5')
        train_data = TaxiDataset('train')
        test_data = TaxiDataset('test')
        print >>sys.stderr, 'done'

    httpd = SocketServer.TCPServer(('', int(sys.argv[1])), VisualizerHTTPRequestHandler)
    httpd.serve_forever()
