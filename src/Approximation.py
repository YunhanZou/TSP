"""
install networkx version 2.2 through terminal:
python -m pip install --upgrade pip
python -m pip install networkx
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys
import time
import os
from Output import Output
from Input import format_check, parse_input, adjacency_mat
import multiprocessing
from threading import Thread, Event


# O(m log n) todo O(n^2)
def computeMST(g):

    # binary heap - O(m log n)
    t = nx.minimum_spanning_tree(g, algorithm='prim')
    assert nx.is_tree(t)

    # O(n)
    w = nx.algorithms.tree.branchings.branching_weight(t)
    # print "mst", w

    return t


# max[O(n^2), O(computeMST)]
def compute(d, input):

    for i in range(d):
        input[i][i] = 0

    m = np.matrix(input)

    # O(n^2)
    g = nx.from_numpy_matrix(m)

    t = computeMST(g)
    assert d == len(t.nodes)

    # O(n * .) = O(n^2)
    for src in range(d):

        # O(m + n)
        gen = nx.dfs_preorder_nodes(t, src)
        tour = list(gen) + [src]

        # O(n)
        c = np.sum([input[tour[i]][tour[i+1]] for i in range(d)])
        # print "tsp", c

        # assert c < 2 * opt

        break

    return c, tour


if __name__ == "__main__":
    """
    # find data_mst -name "*.gr" -exec sed -i '' '1s/^/# /' {} \;

    dir = "../data_mst/"
    for filename in os.listdir(dir):

        if filename.endswith(".gr"):
            # print filename

            g = nx.read_weighted_edgelist(dir + filename, nodetype=int, create_using=nx.MultiGraph())

            start_MST = time.time()
            computeMST(g)
            end_MST = time.time()

            total_time = (end_MST - start_MST) * 1000
            print total_time # , "s"
            
    # O(m log n) --- n = 1000 ---> O(n^2)
    """
    if (False): # todo modify here to switch mode
        dims = []
        rel_times = []
        rel_dists = []

        dir = "DATA/"
        for filename in os.listdir(dir):
            if not filename.endswith("tsp"):
                continue
            filename = dir + filename

            city, dim, edge_weight_type, coord = parse_input(filename)
            adj_mat = adjacency_mat(dim, edge_weight_type, coord)

            k = 10
            cut_off_sec = 1 # todo
            start_MST = time.time()

            for i in range(k):
                thread = Thread(target=compute, args=(dim, adj_mat))
                thread.start()
                thread.join(timeout=cut_off_sec)
                # assert not thread.is_alive()

            end_MST = time.time()
            total_time = (end_MST - start_MST)

            print city
            print total_time / k

            dims.append(dim)
            rel_times.append(total_time / (dim * dim * np.log(dim)))
            rel_dists.append(np.average(adj_mat) / np.std(adj_mat))

        rel_times = np.array(rel_times)
        rel_times /= rel_times[0]

        rel_dists = np.array(rel_dists)
        rel_dists /= rel_dists[0]

        plt.plot(dims, rel_times, 'ro')
        plt.savefig('Approx.png')

    else:
        filename, algorithm, cut_off_sec, random_seed = format_check()
        city, dim, edge_weight_type, coord = parse_input(filename)
        adj_mat = adjacency_mat(dim, edge_weight_type, coord)  # input matrix

        assert algorithm == "Approx"
        dir, filename = filename.split("/")
        assert dir == "DATA"
        filename, suffix = filename.split(".")
        assert suffix == "tsp"
        print filename

        output = Output(filename, algorithm, cut_off_sec)

        start_MST = time.time()
        c, tour = compute(dim, adj_mat) # todo cut_off_sec
        end_MST = time.time()

        output.solution([c] + tour)
        total_time = (end_MST - start_MST)
        output.sol_trace([(total_time, 1)])