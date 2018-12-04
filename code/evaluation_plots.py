"""
Author: Xia Wu
CSE 6140, Fall 2018, Georgia Tech

File to plot evaluation figures for both local search algorithms
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import sys
import time
import os
import bisect


def read_files(seeds, city, algo):
    traces = []
    times = []
    path = 'output/' + str(algo) + '/'
    for seed in seeds:
        f_name = path + str(city) + '_' + str(algo) + '_600_' + str(seed) + '.trace'
        fh = open(f_name, 'r')
        lines = fh.read().splitlines()
        # print lines
        trace, time = [], []
        for i in range(len(lines)):
            line = lines[i].split(',')
            trace.append((float(line[0]), int(line[1])))
            time.append(float(line[0]))
        fh.close()
        traces.append(trace)
        times.append(time)

    # t = np.zeros([len(traces), len(max(traces, key=lambda x: len(x)))])
    # for i, j in enumerate(traces):
    #     t[i][0:len(j)] = j

    return traces, times


# def plot_QRTD(traces):



def main():
    seeds = [12345, 35313, 45631, 90593, 2935812, 2523643, 3496034, 425, 634506, 540963]
    cities = ['Atlanta', 'Berlin', 'Boston', 'Champaign', 'Cincinnati', 'Denver', 'NYC', 'Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState', 'ulysses16', 'UMissouri']
    optimal_vals = [2003763, 0, 0, 0, 277952, 0, 0, 0, 0, 0, 0, 62962, 6859, 0]
    q_star = [0., 0.5, 1., 1.5, 2., 2.5, 3.]  # relative solution quality q*[%]
    # print len(cities), len(optimal_vals)
    # for city in cities:
    #     print read_files(seeds, city, 'LS1')
    traces, times = read_files(seeds, 'Atlanta', 'LS1')
    print "traces: \n, ", traces
    print "times: \n, ", times
    q_star = 0.8 / 100.
    optimal_val = 2003763
    quality = optimal_val * (1 + q_star)  # compute relative solution quality
    print "quality: {}".format(quality)
    best_quals = []
    plot_times = []
    flags = []
    t = 0.
    while t < 1.:  # position of different cut off time
    # for t in range(0, 10, 1):  # position of different cut off time
        plot_times.append(t)
        quals = []
        flag = []
        for i in range(len(traces)):  # for each trace
            ind = bisect.bisect(times[i], t)
            print
            print t, ind, times[i]
            print traces[i]
            # print traces[i][ind]
            if ind > 0 and len(traces[i]) > 1:
                q_largest = traces[i][ind - 1][1]
                quals.append(q_largest)
                if q_largest < quality:  # solved
                    flag.append(1)
                else:
                    flag.append(0)
            else:
                q_largest = traces[i][0][1]
                quals.append(q_largest)
                if q_largest < quality:  # solved
                    flag.append(1)
                else:
                    flag.append(0)
        best_quals.append(quals)
        flags.append(flag)
        t += 0.1

    print
    print "timestamps to plot: "
    print plot_times
    print
    print "qualities: "
    print best_quals
    print
    print "flag: "
    print flags

    p_solve = [flags[i].count(1) / 10. for i in range(len(flags))]
    print
    print "p solve: "
    print p_solve

    plt.plot(plot_times, p_solve)
    plt.savefig('test.png')


if __name__ == '__main__':

    main()
