"""
Author: Xia Wu
CSE 6140, Fall 2018, Georgia Tech

Main file of this project
"""

from Input import format_check, parse_input, adjacency_mat, write_adj_mat_file
from Output import Output
from Approximation import compute
from BranchNBoundImproved import BranchNBound
from IteratedLocalSearch import IteratedLocalSearch as ILS
from SimulatedAnnealing import SimulatedAnnealing as SA
import time


def main():
    filename, algorithm, cut_off_sec, random_seed = format_check()  # check input format
    city, dim, edge_weight_type, coord = parse_input(filename)  # parse input information
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)  # input matrix
    write_adj_mat_file(adj_mat, city, dim)  # save input matrix as csv

    if algorithm == 'Approx':
        output = Output(filename, algorithm, cut_off_sec, algorithm)  # init output object

        start_MST = time.time()
        c, tour = compute(dim, adj_mat, cut_off_sec=cut_off_sec)
        total_time = time.time() - start_MST

        output.solution([c] + tour)  # generate solution file
        output.sol_trace([('%.4f' % total_time, c)])  # generate solution trace file

    elif algorithm == 'BnB':
        output = Output(filename, algorithm, cut_off_sec, algorithm)  # init output object

        bnb = BranchNBound(adj_mat, dim, cut_off_sec)  # param: dist_matrix, num_city, time_limit
        path, cost, trace_list = bnb.run_branch_and_bound()

        output.solution([cost] + path)  # generate solution file
        output.sol_trace(trace_list)  # generate solution trace file

    elif algorithm == 'LS1':  # Iterated LocalSearch
        output = Output(filename, algorithm, cut_off_sec, algorithm, int(random_seed))  # init output object

        ils = ILS(adj_mat, dim, cut_off_sec, random_seed)  # param: dist_matrix, num_city, time_limit, random_seed
        path, cost, trace_list = ils.iterated_local_search()

        output.solution([cost] + path)  # generate solution file
        output.sol_trace(trace_list)  # generate solution trace file

    elif algorithm == 'LS2':  # Simulated Annealing
        output = Output(filename, algorithm, cut_off_sec, algorithm, int(random_seed))  # init output object

        sa = SA(adj_mat, dim, 1e30, 1, 0.999, random_seed, cut_off_sec)
        path, cost, trace_list = sa.run_simulated_annealing()

        output.solution([cost] + path)  # generate solution file
        output.sol_trace(trace_list)  # generate solution trace file


if __name__ == '__main__':
    main()
