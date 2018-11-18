class Node:
    """Class to represent a single node in a search-space tree"""

    def __init__(self, reduced_matrix, cost, path_so_far, visited):
        self.reduced_matrix = reduced_matrix
        self.cost = cost
        self.path_so_far = path_so_far
        self.visited = visited
        self.path_so_far = path_so_far

    def get_reduced_mat(self):
        return self.reduced_matrix

    def get_path_so_far(self):
        return self.path_so_far

    def get_visited(self):
        return self.visited

    def get_cost(self):
        return self.cost
