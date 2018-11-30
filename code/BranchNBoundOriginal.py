import numpy as np
import time
import copy
import math
import sys

from PriorityQueue import PriorityQueue
from Node import Node
from Output import Output
from Input import format_check, parse_input, adjacency_mat


class BranchNBoundOriginal:
    """Class to implement Branch and Bound algorithm"""

    def __init__(self, dist_matrix, num_city, time_limit):
        self.cost_matrix = np.asarray(dist_matrix, dtype='float')
        self.n = num_city
        self.best_path, self.upper_bound = initBestPath(self.cost_matrix)

        print('The initialized path has a cost of ' + str(self.upper_bound))

        # self.start_time = int(round(time.time() * 1000))
        self.time_limit = time_limit * 1000  # time limit in millisec
        # print('self.time_limit = ' + str(self.time_limit))
        self.best_soln_quality = 0.0

        self.preprocess_cost_matrix()  # Set diagonal elements to infinity

        reduced_matrix = np.copy(self.cost_matrix)
        cost, reduced_matrix = reduce_matrix(reduced_matrix)

        self.pq = PriorityQueue()

        visited = set()
        visited.add(0)
        root = Node(reduced_matrix, cost, [0], visited)
        root_node = root.get_cost(), root
        self.pq.append(root_node)

    def preprocess_cost_matrix(self):
        """Set diagonal elements of distance matrix to be infinity"""

        np.fill_diagonal(self.cost_matrix, float('inf'))

    def run_branch_and_bound(self):
        """Body of branch and bound, return the best solution within time limit."""

        start_time = int(round(time.time() * 1000))
        print('The start time is ' + str(start_time) + '. The time limit is ' + str(self.time_limit))
        duration = 0

        while not self.pq.is_empty() and duration < self.time_limit:
            _, content = self.pq.pop()

            cost_so_far = content.get_cost()

            if cost_so_far < self.upper_bound:
                path_so_far = content.get_path_so_far()
                curr_node_idx = path_so_far[-1]  # the node to be expanded
                reduced_matrix = content.get_reduced_mat()
                visited = content.get_visited()

                neighbors = [i for i in range(self.n) if i not in visited]

                # A solution is found
                if np.all(reduced_matrix == float('inf')):
                    print('A solution is found.')
                    if cost_so_far < self.upper_bound:
                        self.upper_bound = cost_so_far
                        print(cost_so_far)
                        self.best_path = path_so_far
                        self.best_soln_quality = duration

                # Branch
                for next_idx in neighbors:
                    reduced_mat_copy = np.copy(reduced_matrix)
                    path_copy = copy.deepcopy(path_so_far)
                    visited_copy = copy.deepcopy(visited)

                    set_row_col_inf(reduced_mat_copy, curr_node_idx, next_idx)
                    reduced_mat_copy[next_idx, 0] = float('inf')  # cannot go back to start point
                    cost, new_reduced_mat = reduce_matrix(reduced_mat_copy)
                    new_cost = cost_so_far + cost + reduced_matrix[curr_node_idx, next_idx]
                    # print('The new_cost is ' + str(new_cost))

                    # Bound
                    if new_cost < self.upper_bound:
                        path_copy.append(next_idx)
                        visited_copy.add(next_idx)
                        content = Node(new_reduced_mat, new_cost, path_copy, visited_copy)
                        next_node = new_cost, content
                        self.pq.append(next_node)

            # Update timer
            curr_time = int(round(time.time() * 1000))
            duration = curr_time - start_time

        self.best_path.append(0)

        return self.best_path, self.upper_bound, self.best_soln_quality/1000.0


def initBestPath(adj_matrix):
    """Greedy method to initialize the best route and upper bound to prune more trees."""

    path = [0]
    cost = 0
    remaining_nodes = set()
    n = adj_matrix.shape[0]

    for i in range(1, n):
        remaining_nodes.add(i)

    curr_node = 0
    next_node = None
    while not len(remaining_nodes) == 0:
        edge_cost = float('inf')
        for neighbor in remaining_nodes:
            next_cost = adj_matrix[curr_node][neighbor]
            if next_cost < edge_cost:
                edge_cost = next_cost
                next_node = neighbor

        path.append(next_node)
        cost += adj_matrix[curr_node][next_node]
        curr_node = next_node
        remaining_nodes.remove(next_node)

    cost += adj_matrix[next_node][0]

    return path, cost


def reduce_matrix(reduced_matrix):
    """
    Perform matrix reduction. Specifically, subtract each row and column with corresponding minimum
    and update cost
    """

    row_mins = np.min(reduced_matrix, axis=1)
    row_mins = row_mins.reshape((-1, 1))
    row_mins[row_mins == float('inf')] = 0

    reduced_matrix = reduced_matrix - row_mins
    col_mins = np.min(reduced_matrix, axis=0)
    col_mins = col_mins.reshape((1, -1))
    col_mins[col_mins == float('inf')] = 0

    reduced_matrix = reduced_matrix - col_mins

    cost = (np.sum(row_mins) + np.sum(col_mins))

    return cost, reduced_matrix


def set_row_col_inf(reduced_matrix, i, j):
    reduced_matrix[i, :] = float('inf')
    reduced_matrix[:, j] = float('inf')
    reduced_matrix[j, i] = float('inf')


def print_path(path):
    print('The shortest TSP path found is: ' + str(path[0])),

    for i in range(1, len(path)):
        print('-> ' + str(path[i])),
    print('\n')


if __name__ == "__main__":

    # cost_matrix = np.array([[0, 20, 30, 10, 11],
    #                         [15, 0, 16, 4, 2],
    #                         [3, 5, 0, 2, 4],
    #                         [19, 6, 18, 0, 3],
    #                         [16, 4, 7, 16, 0]])

    # time_limit = 600  # 10 min
    # bnb1 = BranchNBound(cost_matrix, 5, time_limit)
    # path1, cost1, quality1 = bnb1.run_branch_and_bound()

    # print_path(path1)  # 0 -> 3 -> 1 -> 4 -> 2 -> 0
    # print(cost1)
    # print(quality1)

    # cost_matrix = np.array([[0, 16, 45, 14, 22],
    #                         [16, 0, 18, 11, 4],
    #                         [45, 18, 0, 19, 23],
    #                         [14, 11, 19, 0, 7],
    #                         [22, 4, 23, 7, 0]])

    # bnb2 = BranchNBound(cost_matrix, 5, time_limit)
    # path2, cost2, quality2 = bnb2.run_branch_and_bound()

    # print_path(path2)  # 0 -> 4 -> 1 -> 2 -> 3 -> 0

    # cost_matrix = np.array([[0, 140, 100, 80],
    #                         [140, 0, 90, 69],
    #                         [100, 90, 0, 50],
    #                         [80, 69, 50, 0]])

    # bnb3 = BranchNBound(cost_matrix, 4, time_limit)
    # path3, cost3, quality3 = bnb3.run_branch_and_bound()

    # print_path(path3)  # 0 -> 2 -> 1 -> 3 -> 0

    # cost_matrix = np.array([[0.0, 3.0, 4.0, 2.0, 7.0],
    #                         [3.0, 0.0, 4.0, 6.0, 3.0],
    #                         [4.0, 4.0, 0.0, 5.0, 8.0],
    #                         [2.0, 6.0, 5.0, 0.0, 6.0],
    #                         [7.0, 3.0, 8.0, 6.0, 0.0]])
    # bnb4 = BranchNBound(cost_matrix, 5, time_limit)
    # path4, cost4, quality4 = bnb4.run_branch_and_bound()

    # print_path(path4)
    # print(cost4)

    # cost_matrix = np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]])
    # bnb5 = BranchNBound(cost_matrix, 4, 5)
    # path5, cost5, quality5 = bnb5.run_branch_and_bound()
    # print_path(path5)
    # print(cost5)

    # file_name = "../DATA/Cincinnati.tsp"
    # city, dim, edge_weight_type, coord = parse_input(file_name)
    # adj_mat = adjacency_mat(dim, edge_weight_type, coord)

    # bnb_Atlanta = BranchNBound(adj_mat, dim, 600)

    # path_atl, cost_atl, quality_atl = bnb_Atlanta.run_branch_and_bound()

    filename, algorithm, cut_off_sec, random_seed = format_check()
    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)

    output = Output(filename, algorithm, cut_off_sec)
    bnb = BranchNBoundOriginal(adj_mat, dim, cut_off_sec)
    path, cost, quality = bnb.run_branch_and_bound()

    # print_path(path)
    # print('The cost is ', cost)
    # print('The quality is ', quality)

    output.solution([cost] + path)
    output.sol_trace([(quality, cost)])
