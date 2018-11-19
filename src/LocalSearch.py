import numpy as np
import time
import copy
import math
import sys
import random


class TwoOpt:
	"""
	Class to implement Two-opt Exchange Local Search Algorithm
	"""

	def __init__(self, dist_matrix, num_city, time_limit):
		self.cost_matrix = np.asarray(dist_matrix, dtype='float')
		self.n = num_city
		self.start_time = int(round(time.time() * 1000))
		self.time_limit = time_limit * 60 * 1000
		self.preprocess_cost_matrix()
		self.generate_initial_path()

	def preprocess_cost_matrix(self):
		"""Set diagonal elements of distance matrix to be infinity"""

		np.fill_diagonal(self.cost_matrix, float('inf'))

	def generate_initial_path(self):
		"""
		Generate a random initila path as initial solution.
		Time comlexity: Theta(n)
		Space complexity: Theta(n)
		"""
		not_used = set(range(1, self.n))
		self.path = [0]
		while len(not_used) > 0:
			i = random.sample(not_used, 1)[0]
			self.path.append(i)
			not_used.remove(i)
		self.path.append(0)


def test_initialize():
	cost_matrix = np.array([[0, 20, 30, 10, 11],
												[15, 0, 16, 4, 2],
												[3, 5, 0, 2, 4],
												[19, 6, 18, 0, 3],
												[16, 4, 7, 16, 0]])

	time_limit = 1

	topt1 = TwoOpt(cost_matrix, 5, time_limit)

	print topt1.path
	print topt1.cost_matrix

		
if __name__ == "__main__":
	test_initialize()