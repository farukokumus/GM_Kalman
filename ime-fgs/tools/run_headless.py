#!/usr/bin/env python3
import sys

# matplotlib hack, since we don't want windows
import matplotlib
matplotlib.use('Agg')

# run the other file
filename = sys.argv[1]
fo = open(filename)
exec(compile(fo.read(), filename, 'exec'), globals(), locals())
fo.close()