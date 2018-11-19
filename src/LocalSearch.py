import numpy as np
import time
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
		self.path = []
		not_used = set(range(0, self.n))
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
		Time complexity: Theta(j-i)=>O(n)
		Space complexity: O(1) as inplace
		"""
		while i < j:
			temp = self.path[i]
			self.path[i] = self.path[j]
			self.path[j] = temp
			i += 1
			j -= 1


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

	return topt1

def test_swap():
	topt1 = test_initialize()
	topt1.swap(1, 4)
	print topt1.path



if __name__ == "__main__":
	test_swap()