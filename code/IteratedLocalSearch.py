"""
Author: Henghao Li
CSE 6140, Fall 2018, Georgia Tech

The iterated local search algorithm to solve TSP problem.
"""

import numpy as np
import time
import random
import os

from Output import Output
from Input import format_check, parse_input, adjacency_mat

class TwoOpt:
    """
    Class to implement Two-opt Exchange Local Search Algorithm
    """

    def __init__(self, dist_matrix, num_city, time_limit=6000, random_seed=0):
        """
        Input:
            dist_matrix: np.array that contains distances between cities
            num_city: total number of cities
            time_limit: time_limit in seconds
            random_seed: random seed
        Output: None
        """
        random.seed(random_seed)
        self.dist_matrix = np.asarray(dist_matrix, dtype='float')
        self.n = num_city
        self.start_time = int(round(time.time() * 1000))
        self.time_limit = time_limit * 1000
        self.preprocess_dist_matrix()
        self.generate_initial_path()

    def get_duration(self):
        """
        Input: None
        Output: int duration
        Calculate duration that this algorithm took so far
        """
        curr_time = int(round(time.time() * 1000))
        duration = curr_time - self.start_time
        return duration


    def preprocess_dist_matrix(self):
        """
        Input: None
        Output: None
        Set diagonal elements of distance matrix to be infinity
        """
        np.fill_diagonal(self.dist_matrix, float('inf'))

    def generate_initial_path(self, start_point = 0):
        """
        Input: None
        Output: None
        Generate a random initial path as initial solution.
        Time complexity: Theta(n)
        Space complexity: Theta(n)
        """
        self.path = [start_point]
        not_used = set(range(0, self.n))
        not_used.remove(start_point)
        while len(not_used) > 0:
            i = random.sample(not_used, 1)[0]
            self.path.append(i)
            not_used.remove(i)
        self.path.append(self.path[0])

    def swap(self, i, j):
        """
        Input: int i, int j, j > i
        Basic operation of two-opt
            Breaks two adjacent edges in the path and reconnect after reversing one segment
            Keep path[0~i-1] in original order
            Append path[i~j] in reverse order
            Append path[j+1~] in original order
        This is equivalent to reverse path[i~j] inplace
        Time complexity: Theta(j-i)=>O(n)
        Space complexity: O(1)
        """
        while i < j:
            temp = self.path[i]
            self.path[i] = self.path[j]
            self.path[j] = temp
            i += 1
            j -= 1

    def eval_initial_path(self):
        """
        Evaluate the length of current path.
        Iterative query the distance between adjacent cities in self.cost_matrix and add to the total distance.
        Time complexity: Theta(n)
        Space complexity: O(1)
        """
        dist = 0
        for i in range(0, self.n):
            # print self.dist_matrix[self.path[i]][self.path[i + 1]]
            dist += self.dist_matrix[self.path[i]][self.path[i + 1]]
        return dist

    def eval_try_path(self, old_quality, i, j):
        """
        Input: Old path quality, int i,  int j, j > i
        Output: Tentative path quality if we 2-opt index i and j
        Evaluate the length of path after swap i, j.
        Note that we don't need to re-calculate the entire path; 
        just break the path on i-1~i and j~j+1, and add the length of i-1~j and i~j+1.
        """
        # print old_quality, i, j
        # print self.path
        # self.swap(i, j)
        # print self.path
        # print self.eval_initial_path()
        # self.swap(i, j)
        # print "Is my path correct?"
        # print self.path[i-1], self.path[i], self.dist_matrix[self.path[i-1]][self.path[i]]
        # print self.path[j], self.path[j+1], self.dist_matrix[self.path[j]][self.path[j+1]]
        # print self.path[i-1], self.path[j], self.dist_matrix[self.path[i-1]][self.path[j]]
        # print self.path[i], self.path[j+1], self.dist_matrix[self.path[i]][self.path[j+1]]
        # print old_quality, i, j
        new_quality = old_quality
        new_quality -= self.dist_matrix[self.path[i-1]][self.path[i]]
        new_quality -= self.dist_matrix[self.path[j]][self.path[j+1]]
        new_quality += self.dist_matrix[self.path[i-1]][self.path[j]]
        new_quality += self.dist_matrix[self.path[i]][self.path[j+1]]
        return new_quality

    def two_opt(self):
        """
        Output: best quality (minimum path length found), path, duration of this run 
        Body of two-opt local search algorithm.
        Starting from the begnning of path, iterate through all possible swap and see if an improvement can be made.
        Terminate when no improvement can be made, or no time left.
        Note that this method might get stuck in local minimum.
        """
        duration = self.get_duration()
        best_quality = self.eval_initial_path()
        try_quality = best_quality

        # Stop condition:
        # 1. No improvement can be made in an iteration that goes through the entire path
        # 2. We are running out of time 

        while self.time_limit - duration > 1:
            can_improve = False
            for i in range(1, self.n-1):
                for j in range(i + 1, self.n):
                    try_quality = self.eval_try_path(best_quality, i, j)
                    if try_quality < best_quality:
                        best_quality = try_quality
                        self.swap(i, j)
                        can_improve = True
                        break
                if can_improve:
                    duration = self.get_duration()
                    break
            if not can_improve:
                duration = self.get_duration()
                break

        return self.path, best_quality, duration


class IteratedLocalSearch:
    """
    Wraper of TwoOpt to perform iterated local search using four bridge move
    """
    def __init__(self, dist_matrix, num_city, time_limit=10, random_seed=0):
        """
        Input:
            dist_matrix: np.array that contains distances between cities
            num_city: total number of cities
            time_limit: time_limit in minutes
            random_seed: random seed
        Output: None
        """
        self.twoopt = TwoOpt(dist_matrix, num_city, time_limit, random_seed)
        self.trace_list = []

    def double_bridge_perturbation(self):
        """
        Input: None
        Output: None
        Break current path into four segments using three random chosen points
        Use double bridge move to perturbate current path
        """
        # print self.twoopt.path
        # print self.twoopt.n
        path = self.twoopt.path
        n = self.twoopt.n
        b1 = random.randint(0, n-4)
        b2 = random.randint(b1+1, n-3)
        b3 = random.randint(b2+1, n-2)
        # print b1, b2, b3
        s1 = path[0:b1+1]
        s2 = path[b1+1:b2+1]
        s3 = path[b2+1:b3+1]
        s4 = path[b3+1:-1]
        # print s1, s2, s3, s4
        new_path = s1+s4+s3+s2+[path[0]]
        # print new_path
        self.twoopt.path = new_path

    def iterated_local_search(self, prob=1, decay=0.9999):
        """
        Input:
            prob: flaot, the probability to start next pertubation
            decay: decay rate of probability
        Output:
            path, cost, duration
        Body of iterated local search, which adapts the idea of simulate annealing.
        The probability of "whether I should pertubate and local search once more" is decided by prob.
        For every pertubation that does not improve the solution, decay the probability.
        """
        print("---------------------------------")
        print('Now running Iterated Local Search\n')

        # Perform local search once to get the first best path
        best_path, best_cost, duration = self.twoopt.two_opt()
        timestamp = duration / 1000.0
        self.trace_list.append(('%.4f' % timestamp, best_cost))  # init trace of the first path
        count = 1

        # Iteratively improve using perturbation
        while self.twoopt.time_limit - duration > 1:
            print('Duration: ' + str(duration / 1000.0) + ', time limit: ' + str(self.twoopt.time_limit / 1000.0) + ', shortest distance: ' + str(best_cost))

            if random.random() > prob:
                break
            else:
                count += 1
                # Perturbate and restart local search
                self.double_bridge_perturbation()
                new_path, new_cost, duration = self.twoopt.two_opt()
                if new_cost < best_cost:
                    # Adopt this new best result
                    best_path = new_path
                    best_cost = new_cost
                    duration = self.twoopt.get_duration()
                    timestamp = duration / 1000.0
                    self.trace_list.append(('%.4f' % timestamp, best_cost))  # update trace when new best cost found
                    prob = 1
                else:
                    # Revert to previous best path; decay the probability
                    self.twoopt.path = best_path
                    prob *= decay
                    duration = self.twoopt.get_duration()

        return best_path, best_cost, self.trace_list
        
    
def test_initialize():
    dist_matrix = np.array([[0, 20, 30, 10, 11],
                            [20, 0, 16, 4, 2],
                            [30, 16, 0, 2, 4],
                            [10, 4, 2, 0, 3],
                            [11, 2, 4, 3, 0]])

    time_limit = 1

    topt1 = TwoOpt(dist_matrix, 5, time_limit)

    print topt1.path
    print topt1.dist_matrix

    return topt1


def test_swap():
    topt1 = test_initialize()
    topt1.swap(1, 4)
    print topt1.path


def test_eval_initial_path():
    topt1 = test_initialize()
    print topt1.eval_initial_path()


def test_eval_try_path():
    topt1 = test_initialize()
    old_quality = topt1.eval_initial_path()
    print topt1.eval_try_path(old_quality, 3, 4)


def test_two_opt():
    topt1 = test_initialize()
    print topt1.two_opt()


def test_io():
    filename = "DATA/Cincinnati.tsp"
    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)
    algorithm = "TwoOpt"
    cut_off_sec = 10

    output = Output(filename, algorithm, cut_off_sec)
    to = TwoOpt(adj_mat, dim, cut_off_sec)
    path, cost, quality = to.two_opt()

    output.solution([cost] + path)
    output.sol_trace([(quality, cost)])


def test_ils():
    # filename = "DATA/Cincinnati.tsp"
    filename = "DATA/UKansasState.tsp"

    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)
    algorithm = "IteratedLocalSearch"
    cut_off_sec = 600

    output = Output(filename, algorithm, cut_off_sec)
    ils = IteratedLocalSearch(adj_mat, dim, cut_off_sec, 20)
    path, cost, quality, _ = ils.iterated_local_search()

    output.solution([cost] + path)
    output.sol_trace([(quality, cost)])


def run_all_data():
    dir = "DATA/"
    for f in os.listdir(dir):
        if f[-3:] == "tsp":
            print dir+f
            filename = dir+f
            
            city, dim, edge_weight_type, coord = parse_input(filename)
            adj_mat = adjacency_mat(dim, edge_weight_type, coord)
            algorithm = "IteratedLocalSearch"
            cut_off_sec = 600
            random_seed = 0

            output = Output(filename, algorithm, cut_off_sec)
            ils = IteratedLocalSearch(adj_mat, dim, cut_off_sec, random_seed)
            path, cost, trace_list = ils.iterated_local_search()
            print path, cost, trace_list

            output.solution([cost] + path)
            output.sol_trace(trace_list)

if __name__ == "__main__":
    run_all_data()
