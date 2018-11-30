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
        self.time_limit = time_limit  # in sec

    def run_simulated_annealing(self):
        """Body of the Simulated Annealing algorithm"""

        print("---------------------------------")
        print('Now running Simulated Annealing\n')
        
        # timeout = time.time() + self.time_limit
        counter = 0
        start_time = time.time()
        decrease_T_factor = 1
        duration = 0.0

        while duration < self.time_limit:
            print('Duration: ' + str(duration) + ', time limit: ' + str(self.time_limit) + ', shortest distance: ' + str(self.path_cost))
            T = self.start_T * decrease_T_factor
            self.seed *= 2
            random.seed(self.seed)
            # print('In the current iteration, the initial path cost is ' + str(self.path_cost))
            while T > self.end_T:
                # print T, self.restart_tour_cost
                # if counter % 1000 == 0:
                #     print('After iteration ' + str(counter) + ', the cost is ' + str(self.path_cost))
                a = random.randint(1, self.n - 1)  # don't swap the node at index 0
                b = random.randint(1, self.n - 1)

                start_ind = min(a, b)
                end_ind = max(a, b)

                if end_ind == start_ind:
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
                        # curr_time = int(round(time.time() * 1000))
                        self.quality = time.time() - start_time  # log in sec
                else:
                    diff = self.path_cost - new_dist
                    prob = math.exp(float(diff) / float(T))
                    if prob > random.uniform(0, 1):
                        self.best_soln = copy.deepcopy(new_tour)
                        self.path_cost = new_dist

                T *= self.cooling_factor

            # Restart
            self.best_soln = copy.deepcopy(self.restart_tour)
            new_cost = calculate_init_distance(self.best_soln, self.cost_mat)
            self.path_cost = new_cost

            decrease_T_factor *= 0.8

            time.sleep(1)  # prevent CPU hogging

            duration = time.time() - start_time  # update timer

        return self.best_soln, self.path_cost, self.quality

    def random_tour(self):
        """Generate a random tour at initialization"""

        num_cities = self.n
        tour = [i for i in range(num_cities)]

        random.seed(self.seed)
        random.shuffle(tour)
        tour.append(tour[0])  # last city back to the starting one

        dist = calculate_init_distance(tour, self.cost_mat)

        return tour, dist

    def swap_tour(self, start_ind, end_ind):
        """
        Given the start and end index, reverse the cities between them (including them).

        Cite: http://www.stat.yale.edu/~pollard/Courses/251.spring2013/Handouts/Chang-MoreMC.pdf
        """

        tour = self.best_soln[:]
        tour[start_ind:end_ind+1] = reversed(tour[start_ind:end_ind+1])

        return tour


def calculate_init_distance(tour, adj_matrix):
    """Used to calculate the total distance of a tour at each restart, O(n)"""

    num_cities = adj_matrix.shape[0]

    dist = 0.0
    for i in range(num_cities):
        curr_city = tour[i]
        next_city = tour[i+1]
        dist += adj_matrix[curr_city, next_city]

    return dist


def update_distance(old_distance, old_tour, adj_matrix, start, end):
    """Update distance in O(1)"""

    start_city = old_tour[start]
    end_city = old_tour[end]
    before_start = old_tour[start-1]
    after_end = old_tour[end+1]

    new_distance = old_distance - adj_matrix[before_start][start_city] - adj_matrix[end_city][after_end]
    new_distance += adj_matrix[before_start][end_city] + adj_matrix[start_city][after_end]

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
    sa = SimulatedAnnealing(adj_mat, dim, 1e20, 0.0001, 0.99, 40, 600)
    path, cost, quality = sa.run_simulated_annealing()

    output.solution([cost] + path)
    output.sol_trace([(quality, cost)])
