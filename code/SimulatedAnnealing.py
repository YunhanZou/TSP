import math
import random
import time
import copy

import numpy as np

from Output import Output
from Input import format_check, parse_input, adjacency_mat


class SimulatedAnnealing:

    def __init__(self, cost_mat, dim, start_T, end_T, cooling_factor, seed, time_limit):
        self.cost_mat = np.asarray(cost_mat)
        self.n = dim
        self.seed = seed
        self.start_T = start_T
        self.end_T = end_T
        self.cooling_factor = cooling_factor
        self.best_soln, self.path_cost = self.random_tour()
        self.restart_tour = copy.deepcopy(self.best_soln)  # initial tour for restarting annealing
        self.restart_tour_cost = self.path_cost
        self.quality = 0.0
        self.time_limit = time_limit * 1000  # in millisec

    def run_simulated_annealing(self):
        """Body of the Simulated Annealing algorithm"""

        start_time = int(round(time.time() * 1000))
        counter = 0

        duration = 0

        while duration < self.time_limit:
            T = self.start_T
            print('In the current iteration, the initial path cost is ' + str(self.path_cost))

            while T > self.end_T:
                # if counter % 1000 == 0:
                #     print('After iteration ' + str(counter) + ', the cost is ' + str(self.path_cost))
                a = random.randint(1, self.n - 1)  # don't swap the node at index 0
                b = random.randint(1, self.n - 1)

                start_ind = min(a, b)
                end_ind = max(a, b)

                if end_ind - start_ind <= 2:
                    continue

                counter += 1

                new_tour = self.swap_tour(start_ind, end_ind)

                new_dist = update_distance(self.path_cost, self.best_soln, self.cost_mat, start_ind, end_ind)

                if new_dist < self.path_cost:
                    self.best_soln = copy.deepcopy(new_tour)
                    self.path_cost = new_dist
                    if new_dist < self.restart_tour_cost:
                        self.restart_tour = copy.deepcopy(new_tour)
                        self.restart_tour_cost = new_dist
                    curr_time = int(round(time.time() * 1000))
                    self.quality = (curr_time - start_time) / 1000.0
                else:
                    diff = self.path_cost - new_dist
                    prob = math.exp(diff / T)
                    if prob > random.uniform(0, 1):
                        self.best_soln = copy.deepcopy(new_tour)
                        self.path_cost = new_dist
                        curr_time = int(round(time.time() * 1000))

                T *= self.cooling_factor

            # Restart
            self.best_soln = copy.deepcopy(self.restart_tour)
            new_cost = calculate_init_distance(self.best_soln, self.cost_mat)
            self.path_cost = new_cost

            # Update timer
            curr_time = int(round(time.time() * 1000))
            duration = curr_time - start_time

        self.best_soln.append(self.best_soln[0])  # add start city

        return self.best_soln, self.path_cost, self.quality

    def random_tour(self):
        """Generate a random tour at initialization"""

        num_cities = self.n
        tour = [i for i in range(num_cities)]

        random.seed(self.seed)
        random.shuffle(tour)

        dist = calculate_init_distance(tour, self.cost_mat)

        return tour, dist

    def swap_tour(self, start_ind, end_ind):
        """
        Given the start and end index, reverse the cities between them.

        Cite: http://www.stat.yale.edu/~pollard/Courses/251.spring2013/Handouts/Chang-MoreMC.pdf
        """

        tour = self.best_soln[:]
        if end_ind != len(tour)-1:
            tour[start_ind:end_ind+1] = reversed(tour[start_ind:end_ind+1])
        else:
            tour[start_ind:] = reversed(tour[start_ind:])

        return tour


def calculate_init_distance(tour, adj_matrix):
    """Used to calculate the total distance of a tour at each restart, O(n)"""

    num_cities = len(tour)

    dist = 0
    for i in range(num_cities-1):
        curr_city = tour[i]
        next_city = tour[i+1]
        dist += adj_matrix[curr_city, next_city]

    dist += adj_matrix[tour[-1], tour[0]]  # add dist from last city to start city

    return dist


def update_distance(old_distance, old_tour, adj_matrix, start_ind, end_ind):
    """Update distance in O(1)"""

    if end_ind != len(old_tour)-1:
        new_distance = old_distance - adj_matrix[old_tour[start_ind-1], old_tour[start_ind]] \
                    - adj_matrix[old_tour[end_ind], old_tour[end_ind+1]]
        new_distance += adj_matrix[old_tour[start_ind-1], old_tour[end_ind]] \
                        + adj_matrix[old_tour[start_ind], old_tour[end_ind+1]]
    else:
        new_distance = old_distance - adj_matrix[old_tour[start_ind-1], old_tour[start_ind]] \
                       - adj_matrix[old_tour[end_ind], old_tour[0]]
        new_distance += adj_matrix[old_tour[start_ind-1], old_tour[end_ind]] \
                        + adj_matrix[old_tour[start_ind], old_tour[0]]

    return new_distance


def print_path(input_path, input_cost):
    print('The shortest TSP path found is: ' + str(input_path[0])),

    for i in range(1, len(input_path)):
        print('-> ' + str(input_path[i])),

    print('\nThe total cost is ' + str(input_cost))


if __name__ == "__main__":

    cost_matrix = np.array([[0, 20, 30, 10, 11],
                           [20, 0, 16, 4, 2],
                           [30, 16, 0, 2, 4],
                           [10, 4, 2, 0, 3],
                           [11, 2, 4, 3, 0]])
    sa = SimulatedAnnealing(cost_matrix, 5, 1e5, 0.0001, 0.99, 10, 10)

    path, cost, quality = sa.run_simulated_annealing()

    print_path(path, cost)

    filename, algorithm, cut_off_sec, random_seed = format_check()
    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)

    # output = Output(filename, algorithm, cut_off_sec)
    # sa = SimulatedAnnealing(adj_mat, dim, 1e20, 0.0001, 0.99, 50, 5)
    # path, cost, quality = sa.run_simulated_annealing()

    # output.solution([cost] + path)
    # output.sol_trace([(quality, cost)])
