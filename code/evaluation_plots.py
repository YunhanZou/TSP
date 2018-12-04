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


def main():
    seeds = [12345, 35313, 45631, 90593, 2935812, 2523643, 3496034, 425, 634506, 540963]
    cities = ['Atlanta', 'Berlin', 'Boston', 'Champaign', 'Cincinnati', 'Denver', 'NYC', 'Philadelphia', 'Roanoke', 'SanFrancisco', 'Toronto', 'UKansasState', 'ulysses16', 'UMissouri']
    optimal_vals = [2003763, 7542, 893536, 52643, 277952, 100431, 1555060, 1395981, 655454, 810196, 1176151, 62962, 6859, 132709]
    q_stars = [0., 0.5, 1., 1.5, 2., 2.5, 3.]  # relative solution quality q*[%]

    # cities = ['Boston']
    # optimal_vals = [893536]
    # q_stars = [0., 0.5]  # relative solution quality q*[%]

    for c in range(len(cities)):
        traces, times = read_files(seeds, cities[c], 'LS1')
        optimal_val = optimal_vals[c]
        t_max = max([times[k][-1] for k in range(len(times))])  # maximum times in all trace files of this city
        all_p_solve = []
        print
        print "--QRTD--"
        print "city: ", cities[c]
        print "traces: \n", traces
        print "times: \n", times
        print "max time: \n", t_max
        t_max *= 1.6  # x range of plot
        plt.figure(figsize=(20, 10))
        for q_star_plot in q_stars:
            q_star = q_star_plot / 100.
            accepet_quality = optimal_val * (1 + q_star)  # compute relative solution quality
            # print
            # print "q *: "
            # print q_star
            # print "accepet quality: {}".format(accepet_quality)
            best_quals = []
            plot_times = []
            flags = []
            t = 0.
            while t <= t_max:  # position of different cut off time
                plot_times.append(t)
                quals = []
                flag = []
                for i in range(len(traces)):  # for each trace
                    ind = bisect.bisect(times[i], t)  # find the index of the smallest time that larger than given time t
                    # print
                    # print t, ind, times[i]
                    # print traces[i]
                    # print traces[i][ind]
                    if ind > 0 and len(traces[i]) > 1:
                        q_largest = traces[i][ind - 1][1]
                        quals.append(q_largest)
                        if q_largest <= accepet_quality:  # solved
                            flag.append(1)
                        else:
                            flag.append(0)
                    else:
                        q_largest = traces[i][0][1]
                        quals.append(q_largest)
                        if q_largest <= accepet_quality:  # solved
                            flag.append(1)
                        else:
                            flag.append(0)
                best_quals.append(quals)
                flags.append(flag)
                t += 0.001  # step of time

            # print
            # print "timestamps to plot: "
            # print plot_times
            # print
            # print "qualities: "
            # print best_quals
            # print
            # print "flag: "
            # print flags

            p_solve = [flags[j].count(1) / 10. for j in range(len(flags))]  # compute p_solve for current q*
            all_p_solve.append(p_solve)
            # print
            # print "p solve: "
            # print p_solve
            label = 'q* = ' + str(q_star_plot) + '%'
            plt.plot(plot_times, p_solve, label=label)  # plot line of current p* of this city

        plt.legend()
        plt.title('QRTD of city ' + str(cities[c]))
        plt.xlabel('Run-time (seconds)')
        plt.ylabel('P (Solve)')
        plt.grid(linestyle='dashed', linewidth=1)
        plot_name = 'plots/QRTD/' + str(cities[c]) + '_QRTD.png'
        plt.savefig(plot_name)

        # plot SQD
        print
        print "--SQD--"
        print "city: ", cities[c]
        print len(plot_times)
        plt.figure(figsize=(20, 10))
        t_sqd = len(plot_times) - 1
        step = 4
        while t_sqd < len(plot_times):
            time_plot_sqd = []
            # print t_sqd
            for q in range((len(q_stars))):
                # print plot_times[t_sqd], all_p_solve[q][t_sqd]
                time_plot_sqd.append(all_p_solve[q][t_sqd])
            label_sqd = str(plot_times[t_sqd]) + 's'
            plt.plot(q_stars, time_plot_sqd, label=label_sqd)
            t_sqd = len(plot_times) // pow(2, step)
            step -= 1
        plt.legend()
        plt.title('SQD of city ' + str(cities[c]))
        plt.xlabel('Relative Solution Quality [%]')
        plt.ylabel('P (Solve)')
        plt.grid(linestyle='dashed', linewidth=1)
        plot_name_sqd = 'plots/SQD/' + str(cities[c]) + '_SQD.png'
        plt.savefig(plot_name_sqd)


if __name__ == '__main__':

    main()
