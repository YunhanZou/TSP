import numpy as np
import random
import time
import math


class SimulatedAnnealing:

    def __init__(self, cost_mat, dim, seed):
        self.cost_mat = np.asarray(cost_mat)
        self.n = dim
        self.seed = seed
        self.best_soln, self.path_cost = self.random_tour()
        self.quality = 0.0

    def run_simulated_annealing(self, T, beta):
        """Body of the Simulated Annealing algorithm"""

        start_time = int(round(time.time() * 1000))
        iter = 0

        while T > 0.5:
            iter += 1
            start_ind = random.randint(0, self.n-1)
            end_ind = random.randint(0, self.n-1)

            if start_ind > end_ind:
                temp = start_ind
                start_ind = end_ind
                end_ind = temp

            if start_ind == end_ind:
                continue

            new_tour = self.swap_tour(start_ind, end_ind)

            new_dist = calc_distance(new_tour, self.cost_mat)

            if new_dist <= self.path_cost:
                self.best_soln = new_tour
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

            T *= beta

        self.best_soln.append(self.best_soln[0])  # add start city

        return self.best_soln, self.path_cost, self.quality, iter

    def random_tour(self):
        """Generate a random tour for initialization"""

        num_cities = self.n
        tour = [i for i in range(num_cities)]

        random.seed(self.seed)
        random.shuffle(tour)

        dist = calc_distance(tour, self.cost_mat)

        return tour, dist

    def swap_tour(self, start_ind, end_ind):
        """
        Given the start and end index, reverse the city between them.

        Cite: http://www.stat.yale.edu/~pollard/Courses/251.spring2013/Handouts/Chang-MoreMC.pdf
        """

        tour = self.best_soln[:]
        tour[start_ind:end_ind+1] = reversed(tour[start_ind:end_ind+1])

        return tour


def calc_distance(tour, cost_mat):
    """A helper function to calculate the total distance of current tour."""

    dist = 0
    num_cities = cost_mat.shape[0]
    for i in range(num_cities-1):
        curr_city = tour[i]
        next_city = tour[i+1]
        dist += cost_mat[curr_city, next_city]

    dist += cost_mat[tour[-1], tour[0]]  # add dist from last city to start city

    return dist


def print_path(input_path, input_cost):
    print('The shortest TSP path found is: ' + str(input_path[0])),

    for i in range(1, len(input_path)):
        print('-> ' + str(input_path[i])),

    print('\n The total cost is ' + str(input_cost))


if __name__ == "__main__":
    """Debugging purpose"""

    cost_matrix = np.array([[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]])
    sa = SimulatedAnnealing(cost_matrix, 4, 50)

    path, cost, quality, iteration = sa.run_simulated_annealing(10, 0.99)

    print_path(path, cost)

    cost_matrix = np.array([[0, 20, 30, 10, 11],
                            [15, 0, 16, 4, 2],
                            [3, 5, 0, 2, 4],
                            [19, 6, 18, 0, 3],
                            [16, 4, 7, 16, 0]])
    sa = SimulatedAnnealing(cost_matrix, 5, 50)

    path, cost, quality, iteration = sa.run_simulated_annealing(1000, 0.99)

    print_path(path, cost)
