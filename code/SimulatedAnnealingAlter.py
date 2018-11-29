import numpy as np
import time
import random
import os
import math

from Output import Output
from Input import format_check, parse_input, adjacency_mat

class SimulatedAnnealing:
    """
    Class to implement Two-opt Exchange Local Search Algorithm
    """

    def __init__(self, dist_matrix, num_city, start_T, end_T, cooling_factor, random_seed = 0, time_limit = 6000):
        """
        Input:
            dist_matrix: np.array that contains distances between cities
            num_city: total number of cities
            time_limit: time_limit in second
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
        self.start_T = start_T
        self.end_T = end_T
        self.cooling_factor = cooling_factor
        

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

    def simulated_annealing(self):
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
        T = self.start_T

        # Stop condition:
        # 1. No improvement can be made in an iteration that goes through the entire path
        # 2. We are running out of time 

        while self.time_limit - duration > 1 and T > self.end_T:
            print T, best_quality
            i = random.randint(1, self.n-2) 
            j = random.randint(i+1, self.n-1) 
            try_quality = self.eval_try_path(best_quality, i, j)
            if try_quality < best_quality:
                best_quality = try_quality
                self.swap(i, j)
            else: #Accept with certain probability
                diff =  best_quality - try_quality
                # print diff, T, diff / T, math.exp(diff / T)
                prob = math.exp(diff / T)
                if prob > random.uniform(0, 1):
                    best_quality = try_quality
                    self.swap(i, j)
                    can_improve = True
            T *= self.cooling_factor

        return self.path, best_quality, duration


def run_all_data():
    dir = "DATA/"
    for f in os.listdir(dir):
        if f[-3:] == "tsp":
            print dir+f
            filename = dir+f
            
            city, dim, edge_weight_type, coord = parse_input(filename)
            adj_mat = adjacency_mat(dim, edge_weight_type, coord)
            algorithm = "SAAlter"
            cut_off_sec = 600
            random_seed = 0

            output = Output(filename, algorithm, cut_off_sec)
            sa = SimulatedAnnealing(adj_mat, dim, 1e20, 0.0001, 0.99, cut_off_sec, random_seed)
            path, cost, quality = sa.simulated_annealing()
            print path, cost, quality

            output.solution([cost] + path)
            output.sol_trace([(quality, cost)])

if __name__ == "__main__":
    run_all_data()
