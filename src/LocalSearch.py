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

	def generate_initial_path(self):
		"""
		Input: None
		Output: None
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

	def eval_path(self):
		"""
		Evaluate the lenght of current path.
		Iterative query the distance between adjacent cities in self.cost_matrix and add to the total distance.
		Time complexity: Theta(n)
		Space complexity: O(1)
		"""
		dist = 0
		for i in range(0, self.n):
			print self.dist_matrix[self.path[i]][self.path[i+1]]
			dist += self.dist_matrix[self.path[i]][self.path[i+1]]
		return dist
			

	def local_search(self):
		curr_time = int(round(time.time() * 1000))
		duration = curr_time - self.start_time


def test_initialize():
	dist_matrix = np.array([[0, 20, 30, 10, 11],
												[15, 0, 16, 4, 2],
												[3, 5, 0, 2, 4],
												[19, 6, 18, 0, 3],
												[16, 4, 7, 16, 0]])

	time_limit = 1

	topt1 = TwoOpt(dist_matrix, 5, time_limit)

	print topt1.path
	print topt1.dist_matrix

	return topt1

def test_swap():
	topt1 = test_initialize()
	topt1.swap(1, 4)
	print topt1.path

def test_eval_path():
	topt1 = test_initialize()	
	print topt1.eval_path()


if __name__ == "__main__":
	print test_eval_path()