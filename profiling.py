from __future__ import division
from cProfile import run
from data_ilsvrc import main
import pstats

filename = 'profiling.stats'
run('main()',filename)
stats = pstats.Stats(filename)
stats.sort_stats('time')
stats.print_stats()






















