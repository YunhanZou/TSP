"""
Author: Yunhan Zou, Henghao Li
CSE 6140, Fall 2018, Georgia Tech

The branch and bound algorithm to solve TSP problem.
"""

import numpy as np
import time
import copy

from PriorityQueue import PriorityQueue
from Node import Node
from Output import Output
from Input import format_check, parse_input, adjacency_mat


class BranchNBound:
    """Class to implement Branch and Bound algorithm"""

    def __init__(self, dist_matrix, num_city, time_limit):
        print('---------------------------------')
        print('Running improved Branch and Bound\n')

        self.cost_matrix = np.asarray(dist_matrix, dtype='float')
        self.n = num_city
        start_time = time.time()
        self.best_path, self.upper_bound = initiate_greedy_path(self.cost_matrix)
        self.best_soln_quality = time.time() - start_time
        self.trace_list = [('%.4f' % self.best_soln_quality, self.upper_bound)]

        self.time_limit = time_limit  # time limit in sec

        self.preprocess_cost_matrix()  # set diagonal elements to infinity

        reduced_matrix = np.copy(self.cost_matrix)
        cost, reduced_matrix = reduce_matrix(reduced_matrix)

        # Initiate a priority queue for each length of partial solution.
        self.pqs = []
        for i in range(0, num_city):
            self.pqs.append(PriorityQueue())

        visited = set()
        visited.add(0)
        root = Node(reduced_matrix, cost, [0], visited)
        root_node = root.get_cost(), root
        self.pqs[0].append(root_node)

    def preprocess_cost_matrix(self):
        """Set diagonal elements of distance matrix to be infinity"""

        np.fill_diagonal(self.cost_matrix, float('inf'))

    def run_branch_and_bound(self):
        """Body of branch and bound, return the best solution within time limit."""

        start_time = time.time()
        duration = self.best_soln_quality
        current_level = 0

        while not (all(pq.is_empty() for pq in self.pqs)) and duration < self.time_limit:
            exausted = False
            entry_level = current_level
            while self.pqs[current_level].is_empty():
                current_level += 1
                if current_level == self.n:
                    current_level = 0
                if current_level == entry_level:
                    exausted = True
                    break
            if exausted:
                break

            if self.pqs[current_level].size() > 1024:
                self.pqs[current_level].queue = self.pqs[current_level].queue[0:1023]

            # print self.upper_bound
            _, content = self.pqs[current_level].pop()

            cost_so_far = content.get_cost()

            if cost_so_far >= self.upper_bound:
                # The shortest path in pq on this level has a lower bound larger than the upper bound
                # so we can prune all the partial solutions on this level
                self.pqs[current_level].clear()

            else:
                path_so_far = content.get_path_so_far()
                curr_node_idx = path_so_far[-1]  # the node to be expanded
                reduced_matrix = content.get_reduced_mat()
                visited = content.get_visited()

                neighbors = [i for i in range(self.n) if i not in visited]

                # A solution is found
                if current_level == self.n-1:
                    self.upper_bound = cost_so_far
                    self.best_path = path_so_far
                    self.best_soln_quality = duration
                    current_level = 0
                    self.trace_list.append(('%.4f' % self.best_soln_quality, self.upper_bound))

                else:
                    # Branch
                    current_level += 1
                    for next_idx in neighbors:
                        reduced_mat_copy = np.copy(reduced_matrix)
                        path_copy = copy.deepcopy(path_so_far)
                        visited_copy = copy.deepcopy(visited)

                        set_row_col_inf(reduced_mat_copy, curr_node_idx, next_idx)
                        reduced_mat_copy[next_idx, 0] = float('inf')  # cannot go back to start point
                        cost, new_reduced_mat = reduce_matrix(reduced_mat_copy)
                        new_cost = cost_so_far + cost + reduced_matrix[curr_node_idx, next_idx]

                        # Bound
                        if new_cost < self.upper_bound:
                            path_copy.append(next_idx)
                            visited_copy.add(next_idx)
                            content = Node(new_reduced_mat, new_cost, path_copy, visited_copy)
                            next_node = new_cost, content
                            self.pqs[current_level].append(next_node)

            duration = time.time() - start_time  # update timer

        self.best_path.append(0)  # append the start city

        print('Duration: ' + str(self.best_soln_quality) + ', time limit: ' + str(self.time_limit) + ', new distance: ' + str(self.upper_bound))
        print('BnB complete.')

        return self.best_path, self.upper_bound, self.trace_list


def initiate_greedy_path(adj_matrix):
    """Greedy method to initialize the best route so more trees can be pruned."""

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
    """Given start and end index i and j, set i-th row, j-th col and [j,i] to inf, indicating visited"""

    reduced_matrix[i, :] = float('inf')
    reduced_matrix[:, j] = float('inf')
    reduced_matrix[j, i] = float('inf')


# For debugging
def print_path(input_path):
    print('The shortest TSP path found is: ' + str(input_path[0])),

    for i in range(1, len(input_path)):
        print('-> ' + str(input_path[i])),
    print('\n')


if __name__ == "__main__":
    """Main function"""

    filename, algorithm, cut_off_sec, random_seed = format_check()
    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)

    output = Output(filename, algorithm, cut_off_sec)
    bnb = BranchNBound(adj_mat, dim, cut_off_sec)
    path, cost, quality = bnb.run_branch_and_bound()

    output.solution([cost] + path)  # generate solution file
    output.sol_trace(quality)  # generate solution trace file
