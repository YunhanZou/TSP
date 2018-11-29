from Input import format_check, parse_input, adjacency_mat, write_adj_mat_file
from Output import Output
from Approximation import compute
from BranchNBound import BranchNBound
from LocalSearch import TwoOpt
import time


def main():
    filename, algorithm, cut_off_sec, random_seed = format_check()  # check input format
    city, dim, edge_weight_type, coord = parse_input(filename)  # parse input information
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)  # input matrix
    write_adj_mat_file(adj_mat, city, dim)  # save input matrix as csv

    output = Output(filename, algorithm, cut_off_sec)  # init output object

    if algorithm == 'Approx':
        start_MST = time.time()
        c, tour = compute(dim, adj_mat)  # TODO: add cut_off_sec
        total_time = time.time() - start_MST

        output.solution([c] + tour)  # generate solution file
        output.sol_trace([(total_time, 1)])  # generate solution trace file

    elif algorithm == 'BnB':
        bnb = BranchNBound(adj_mat, dim, cut_off_sec)  # param: dist_matrix, num_city, time_limit
        path, cost, quality = bnb.run_branch_and_bound()

        output.solution([cost] + path)  # generate solution file
        output.sol_trace([(quality, cost)])  # generate solution trace file

    elif algorithm == 'LS1':
        ls1 = TwoOpt(adj_mat, dim, cut_off_sec)  # param: dist_matrix, num_city, time_limit
        path = ls1.path

        best_quality = 1000000  # TODO: need modification here
        output.solution([best_quality] + path)  # generate solution file

    elif algorithm == 'SA':



if __name__ == '__main__':
    main()
