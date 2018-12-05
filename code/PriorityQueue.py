"""
Author: Yunhan Zou
CSE 6140, Fall 2018, Georgia Tech

A queue structure implemented using heapq where each element is served in order of priority.

Used to store node object, and define priority to be their cost_so_far.

Each tuple stored in the queue has (priority, count, node), and node contains node information.
"""

import heapq


class PriorityQueue:

    def __init__(self):
        self.queue = []
        self.current = 0

    def pop(self):
        """
        Pop a node from the queue.

        """

        (_, _, node) = heapq.heappop(self.queue)
        return node

    def append(self, node):
        """Append a node to the queue"""

        priority, content = node
        entry = priority, self.current, node
        heapq.heappush(self.queue, entry)
        self.current += 1

    def node_list(self):
        """Print the list of nodes currently in the queue"""

        nodes = []
        for idx, (priority, count, node) in enumerate(self.queue):
            _, (_, curr_node) = node
            nodes.append(curr_node)

        return nodes

    def size(self):

        return len(self.queue)

    def clear(self):

        self.queue = []

    def top(self):

        return self.queue[0]

    def is_empty(self):

        return self.size() == 0
