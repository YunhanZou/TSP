import math
import random
import time

import numpy as np

from Output import Output
from Input import format_check, parse_input, adjacency_mat


class SimulatedAnnealing:

    def __init__(self, cost_mat, dim, start_T, end_T, cooling_factor, iter, seed):
        self.cost_mat = np.asarray(cost_mat)
        self.n = dim
        self.seed = seed
        self.start_T = start_T
        self.end_T = end_T
        self.cooling_factor = cooling_factor
        self.num_iter = iter
        self.best_soln, self.path_cost = self.random_tour()
        self.restart_tour = self.best_soln  # initial tour for restarting annealing
        self.quality = 0.0

    def run_simulated_annealing(self):
        """Body of the Simulated Annealing algorithm"""

        start_time = int(round(time.time() * 1000))
        counter = 0

        for i in range(self.num_iter):
            T = self.start_T
            while T > self.end_T:
                if counter % 1000 == 0:
                    print('After iteration ' + str(counter) + ', the cost is ' + str(self.path_cost))
                start_ind = random.randint(0, self.n - 1)
                end_ind = random.randint(0, self.n - 1)

                if abs(end_ind-start_ind) <= 1:
                    continue

                counter += 1

                if start_ind > end_ind:
                    temp = start_ind
                    start_ind = end_ind
                    end_ind = temp

                new_tour = self.swap_tour(start_ind, end_ind)

                new_dist = update_distance(self.path_cost, self.best_soln, self.cost_mat, start_ind, end_ind)

                if new_dist <= self.path_cost:
                    self.best_soln = new_tour
                    self.restart_tour = new_tour
                    self.path_cost = new_dist
                    curr_time = int(round(time.time() * 1000))
                    self.quality = (curr_time - start_time) / 1000.0
                else:
                    diff = self.path_cost - new_dist
                    prob = math.exp(diff / T)
                    if prob > random.uniform(0, 1):
                        self.best_soln = new_tour
                        self.path_cost = new_dist
                        curr_time = int(round(time.time() * 1000))
                        self.quality = (curr_time - start_time) / 1000.0

                T *= self.cooling_factor

            # Restart
            self.best_soln = self.restart_tour

        self.best_soln.append(self.best_soln[0])  # add start city

        return self.best_soln, self.path_cost, self.quality

    def random_tour(self):
        """Generate a random tour for initialization"""

        num_cities = self.n
        tour = [i for i in range(num_cities)]

        random.seed(self.seed)
        random.shuffle(tour)

        dist = 0
        for i in range(num_cities-1):
            curr_city = tour[i]
            next_city = tour[i+1]
            dist += self.cost_mat[curr_city, next_city]

        dist += self.cost_mat[tour[-1], tour[0]]  # add dist from last city to start city

        return tour, dist

    def swap_tour(self, start_ind, end_ind):
        """
        Given the start and end index, reverse the city between them.

        Cite: http://www.stat.yale.edu/~pollard/Courses/251.spring2013/Handouts/Chang-MoreMC.pdf
        """

        tour = self.best_soln[:]
        tour[start_ind+1:end_ind] = reversed(tour[start_ind+1:end_ind])

        return tour


def update_distance(old_distance, old_tour, adj_matrix, start_ind, end_ind):
    """Update distance in O(1)"""

    new_distance = old_distance - adj_matrix[old_tour[start_ind], old_tour[start_ind+1]] - adj_matrix[old_tour[end_ind-1], old_tour[end_ind]]
    new_distance += adj_matrix[old_tour[start_ind], old_tour[end_ind-1]] + adj_matrix[old_tour[start_ind+1], old_tour[end_ind]]

    return new_distance


def print_path(input_path, input_cost):
    print('The shortest TSP path found is: ' + str(input_path[0])),

    for i in range(1, len(input_path)):
        print('-> ' + str(input_path[i])),

    print('\nThe total cost is ' + str(input_cost))


if __name__ == "__main__":

    # cost_matrix = np.array([[0, 20, 30, 10, 11],
    #                        [20, 0, 16, 4, 2],
    #                        [30, 16, 0, 2, 4],
    #                        [10, 4, 2, 0, 3],
    #                        [11, 2, 4, 3, 0]])
    # sa = SimulatedAnnealing(cost_matrix, 5, 1e5, 0.0001, 0.97, 10, 666)

    # path, cost, quality = sa.run_simulated_annealing()

    # print_path(path, cost)

    filename, algorithm, cut_off_sec, random_seed = format_check()
    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)

    output = Output(filename, algorithm, cut_off_sec)
    sa = SimulatedAnnealing(adj_mat, dim, 1e10, 0.0001, 0.97, 10, 666)
    path, cost, quality = sa.run_simulated_annealing()

    output.solution([cost] + path)
    output.sol_trace([(quality, cost)])
