from __future__ import division
from cProfile import run
from ilsvrc import main
import pstats

filename = 'foo.stats'
run('main()',filename)
stats = pstats.Stats(filename)
stats.sort_stats('time')
stats.print_stats()






















