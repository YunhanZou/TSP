"""
Author: Xia Wu
CSE 6140, Fall 2018, Georgia Tech

Parse Input tsp files
"""

import sys
import getopt
import math


def format_check():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "i:a:t:s:", ["inst=", "alg=", "time=", "seed="])
        # print opts, args
        num_args = len(opts)
        if num_args != 3 and num_args != 4:
            print "Please give correct number of input arguments"
            sys.exit(2)
    except getopt.GetoptError:
        print 'format: \nInput.py -inst <filename> -alg [BnB | Approx | LS1 | LS2] -time <cutoff_in_seconds> [-seed <random_seed>]'
        sys.exit(2)

    algo_list = ["BnB", "Approx", "LS1", "LS2"]
    filename, algorithm, cut_off_sec, random_seed = None, None, None, None
    for opt, arg in opts:
        if opt in ("-i", "--inst"):
            dir, file = arg.split("/")
            assert dir == "DATA", "Please provide correct path"
            inst, suffix = file.split(".")
            assert suffix == "tsp", "Please provide correct tsp file"
            filename = arg
        elif opt in ("-a", "--alg"):
            if arg not in algo_list:
                print "Please provide correct algorithm"
                sys.exit(2)
            # print arg
            algorithm = arg
        elif opt in ("-t", "--time"):
            if not arg.isdigit():
                print "Please provide correct cut off seconds"
                sys.exit(2)
            cut_off_sec = arg
        elif opt in ("-s", "--seed"):
            random_seed = arg
            if algorithm == 'BnB' or algorithm == 'Approx':
                if random_seed is not None:
                    print "Please provide correct input arguments"
                    sys.exit(2)
                else:
                    random_seed = None

    # print filename, algorithm, cut_off_sec, random_seed
    if not filename or not algorithm or not cut_off_sec:
        print 'format: \nInput.py -inst <filename> -alg [BnB | Approx | LS1 | LS2] -time <cutoff_in_seconds> [-seed <random_seed>]'
        sys.exit(2)

    return filename, algorithm, float(cut_off_sec), random_seed


def parse_input(f_name):
    try:
        fh = open(f_name, 'r')
    except IOError:
        print "Please provide correct path for input file"
        sys.exit(2)

    lines = fh.read().splitlines()
    # print lines
    dim = 0  # dimension
    city = ''  # city name
    content = []  # coordinates
    edge_weight_type = 'EUC_2D'

    for i in range(len(lines)):
        if lines[i] == 'NODE_COORD_SECTION':  # end of head lines
            content = lines[i + 1: i + dim + 1]
            break
        head, val = lines[i].split(':')
        # print head, val
        if head == 'NAME':
            if val.lstrip() == 'ulysses16.tsp':
                city = 'ulysses'
            else:
                city = val.lstrip()
        elif head == 'DIMENSION':
            dim = int(val.lstrip())
        elif head == 'EDGE_WEIGHT_TYPE':
            edge_weight_type = val.lstrip()

    coord = [(0, 0) for i in range(dim)]  # Note actual index starts at 1
    for line in content:
        l = line.strip()  # remove trailing and leading spaces
        idx, x, y = l.split(' ')
        idx = int(idx) - 1  # index of matrix start from 0
        coord[idx] = (float(x), float(y))

    fh.close()

    return city, dim, edge_weight_type, coord


def euc_2d(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    xd = x1 - x2
    yd = y1 - y2
    dist = math.sqrt(xd * xd + yd * yd)

    return int(round(dist))  # round to nearest integer


def covert_radian(x, y):
    PI = 3.141592
    deg_x, deg_y = int(x), int(y)
    min_x, min_y = x - deg_x, y - deg_y
    latitude = PI * (deg_x + 5.0 * min_x / 3.) / 180.
    longitude = PI * (deg_y + 5.0 * min_y / 3.) / 180.

    return latitude, longitude


def geo(coord1, coord2):
    lat1, long1 = coord1
    lat2, long2 = coord2

    RRR = 6378.388
    q1 = math.cos(long1 - long2)
    q2 = math.cos(lat1 - lat2)
    q3 = math.cos(lat1 + lat2)
    dist = int(RRR * math.acos(0.5 * ((1. + q1) * q2 - (1. - q1) * q3)) + 1.)

    return dist


def adjacency_mat(dim, edge_weight_type, coord):
    adj_mat = [[0 for i in range(dim)] for j in range(dim)]
    if edge_weight_type == 'EUC_2D':
        for i in range(len(coord)):
            for j in range(len(coord)):
                if j >= i:
                    adj_mat[i][j] = euc_2d(coord[i], coord[j])
                else:
                    adj_mat[i][j] = adj_mat[j][i]
                # print adj_mat[i][j]
            # print
            # print adj_mat[i]
    elif edge_weight_type == 'GEO':
        randian = coord[:]
        for i in range(len(randian)):
            randian[i] = covert_radian(randian[i][0], randian[i][1])
        for i in range(len(randian)):
            for j in range(len(randian)):
                if j >= i:
                    adj_mat[i][j] = geo(randian[i], randian[j])
                else:
                    adj_mat[i][j] = adj_mat[j][i]

    return adj_mat


def write_adj_mat_file(adj_mat, city, dim):
    path = 'input/'
    name = 'mat_' + str(city) + '_' + str(dim) + '.csv'
    f = open(path + name, 'w')
    for l in adj_mat:
        line = ','.join(str(l[i]) for i in range(dim)) + '\n'
        f.write(line)
    f.close()


# Usage example
def main():
    filename, algorithm, cut_off_sec, random_seed = format_check()
    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)  # input matrix


if __name__ == '__main__':
    main()
