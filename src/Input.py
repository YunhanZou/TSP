import sys
import getopt
import math

import Approximation, BranchAndBound


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
    # print filename, algorithm, cut_off_sec, random_seed
    if not filename or not algorithm or not cut_off_sec:
        print 'format: \nInput.py -inst <filename> -alg [BnB | Approx | LS1 | LS2] -time <cutoff_in_seconds> [-seed <random_seed>]'
        sys.exit(2)

    return filename, algorithm, cut_off_sec, random_seed


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
            city = val.lstrip()
        elif head == 'DIMENSION':
            dim = int(val.lstrip())
        elif head == 'EDGE_WEIGHT_TYPE':
            edge_weight_type = val.lstrip()
    # print len(content)
    # print city, dim
    # print content
    coord = [(0, 0) for i in range(dim)]  # Note actual index starts at 1
    for line in content:
        l = line.strip()  # remove trailing and leading spaces
        idx, x, y = l.split(' ')
        idx = int(idx) - 1  # index of matrix start from 0
        coord[idx] = (float(x), float(y))
    # print coord

    fh.close()

    return city, dim, edge_weight_type, coord


def euc_2d(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    xd = x1 - x2
    yd = y1 - y2
    dist = math.sqrt(xd * xd + yd * yd)
    # print dist
    # print int(round(dist))
    return int(round(dist))  # round to nearest integer


def covert_radian(x, y):
    PI = 3.141592653
    deg_x, deg_y = int(x), int(y)
    # print deg_x, deg_y
    min_x, min_y = x - deg_x, y - deg_y
    # print min_x, min_y
    latitude = PI * (deg_x + 5.0 * min_x / 30.) / 180.
    longitude = PI * (deg_y + 5.0 * min_y / 30.) / 180.

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
        # print randian
        for i in range(len(randian)):
            randian[i] = covert_radian(randian[i][0], randian[i][1])
        for i in range(len(randian)):
            for j in range(len(randian)):
                if j >= i:
                    adj_mat[i][j] = geo(randian[i], randian[j])
                else:
                    adj_mat[i][j] = adj_mat[j][i]
                # print adj_mat[i][j]
            # print
            # print adj_mat[i]

    return adj_mat


def main():
    filename, algorithm, cut_off_sec, random_seed = format_check()
    city, dim, edge_weight_type, coord = parse_input(filename)
    adj_mat = adjacency_mat(dim, edge_weight_type, coord)  # input matrix


if __name__ == '__main__':
    main()
