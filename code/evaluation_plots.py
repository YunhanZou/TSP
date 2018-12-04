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
import csv


def read_files(seeds, city, algo):
    traces = []
    times = []
    path = 'output/' + str(algo) + '/'
    for seed in seeds:
        f_name = path + str(city) + '_' + str(algo) + '_600_' + str(seed) + '.trace'
        fh = open(f_name, 'r')
        lines = fh.read().splitlines()
        trace, time = [], []
        for i in range(len(lines)):
            line = lines[i].split(',')
            trace.append((float(line[0]), int(line[1])))
            time.append(float(line[0]))
        fh.close()
        traces.append(trace)
        times.append(time)

    return traces, times


def box_plots_data(seeds, cities, optimal_vals, q_stars, algo):
    cities_abbr = []
    times_boxplots = []
    for c in range(len(cities)):
        print
        print "city: ", cities[c]
        if algo =='LS2' and cities[c] == 'Roanoke':
            traces, times = read_files(seeds, cities[c], algo)
            optimal_val = optimal_vals[c]
            q_star = q_stars[-1]
            accept_quality = optimal_val * (1 + q_star)
            times_boxplot = []
            for trace in traces:
                for i in range(len(trace)):
                    if trace[i][1] <= accept_quality:
                        times_boxplot.append([float(trace[i][0])])
                        break
            print len(times_boxplot)
            print times_boxplot
            with open("plots/box_plot/boxplot_data_" + str(algo) + "_Roanoke.csv", "wb") as f:
                writer = csv.writer(f)
                writer.writerows([['Roanoke']])
                writer.writerows(times_boxplot)

            continue

        cities_abbr.append(cities[c][:3])
        traces, times = read_files(seeds, cities[c], algo)

        optimal_val = optimal_vals[c]
        q_star = q_stars[-1]
        accept_quality = optimal_val * (1 + q_star)
        times_boxplot = []
        for trace in traces:
            for i in range(len(trace)):
                if trace[i][1] <= accept_quality:
                    times_boxplot.append(float(trace[i][0]))
                    break
        print len(times_boxplot)
        print times_boxplot
        times_boxplots.append(times_boxplot)

    t_times_boxplots = zip(*times_boxplots)  # transpose
    # print len(t_times_boxplots), len(t_times_boxplots[0])

    with open("plots/box_plot/boxplot_data_" + str(algo) + '.csv', "wb") as f:
        writer = csv.writer(f)
        writer.writerows([cities_abbr])
        writer.writerows(t_times_boxplots)


def qrtd_sqd_plot(seeds, cities, optimal_vals, q_stars, algo):
    for c in range(len(cities)):
        traces, times = read_files(seeds, cities[c], algo)
        optimal_val = optimal_vals[c]
        max_all_times = [times[k][-1] for k in range(len(times))]
        t_max = max(max_all_times)  # maximum times in all trace files of this city

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
            accept_quality = optimal_val * (1 + q_star)  # compute relative solution quality
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
                    if ind > 0 and len(traces[i]) > 1:
                        q_largest = traces[i][ind - 1][1]
                        quals.append(q_largest)
                        if q_largest <= accept_quality:  # solved
                            flag.append(1)
                        else:
                            flag.append(0)
                    else:
                        q_largest = traces[i][0][1]
                        quals.append(q_largest)
                        if q_largest <= accept_quality:  # solved
                            flag.append(1)
                        else:
                            flag.append(0)
                best_quals.append(quals)
                flags.append(flag)
                t += 0.001  # step of time
            #
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
        plt.title('QRTD of city ' + str(cities[c]) + 'with algorithm' + str(algo))
        plt.xlabel('Run-time (seconds)')
        plt.ylabel('P (Solve)')
        plt.grid(linestyle='dashed', linewidth=1)
        plot_name = 'plots/QRTD/' + str(cities[c]) + '_' + str(algo) + '_QRTD.png'
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
        plt.title('SQD of city ' + str(cities[c]) + 'with algorithm' + str(algo))
        plt.xlabel('Relative Solution Quality [%]')
        plt.ylabel('P (Solve)')
        plt.grid(linestyle='dashed', linewidth=1)
        plot_name_sqd = 'plots/SQD/' + str(cities[c]) + '_' + str(algo) + '_SQD.png'
        plt.savefig(plot_name_sqd)


def main():
    seeds = [12345, 35313, 45631, 90593, 2935812, 2523643, 3496034, 425, 634506, 540963]
    cities = ['Cincinnati', 'UKansasState', 'ulysses16', 'Atlanta', 'Philadelphia', 'Boston', 'Berlin', 'Champaign', 'NYC', 'Denver', 'SanFrancisco', 'UMissouri', 'Toronto', 'Roanoke']
    optimal_vals = [277952, 62962, 6859, 2003763, 1395981, 893536, 7542, 52643, 1555060, 100431, 810196, 132709, 1176151, 655454]
    q_stars = [0., 0.5, 1., 1.5, 2., 2.5, 3.]  # relative solution quality q*[%]

    box_plots_data(seeds, cities, optimal_vals, q_stars, 'LS1')
    box_plots_data(seeds, cities, optimal_vals, q_stars, 'LS2')

    qrtd_sqd_plot(seeds, cities, optimal_vals, q_stars, 'LS1')
    qrtd_sqd_plot(seeds, cities, optimal_vals, q_stars, 'LS2')


if __name__ == '__main__':
    main()
