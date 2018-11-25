import numpy as np
import time
import random


class TwoOpt:
    """
    Class to implement Two-opt Exchange Local Search Algorithm
    """

    def __init__(self, dist_matrix, num_city, time_limit):
        """
        Input:
            dist_matrix: np.array that contains distances between cities
            num_city: total number of cities
            time_limit: time_limit
        Output: None
        """
        self.dist_matrix = np.asarray(dist_matrix, dtype='float')
        self.n = num_city
        self.start_time = int(round(time.time() * 1000))
        self.time_limit = time_limit * 60 * 1000
        self.preprocess_dist_matrix()
        self.generate_initial_path()

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
        curr_time = int(round(time.time() * 1000))
        duration = curr_time - self.start_time
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
                    break
            if not can_improve:
                break

        return best_quality, self.path, duration

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


if __name__ == "__main__":
    test_two_opt()
