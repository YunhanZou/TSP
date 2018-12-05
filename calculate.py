"""
install networkx version 2.2 through terminal:
python -m pip install --upgrade pip
python -m pip install networkx
"""
import os
import re
import numpy as np

if __name__ == "__main__":

        dir = "DATA/"
        alg = "LS1"
        filenames = set()
        for filename in os.listdir(dir):
            filenames.add(filename.split(".")[0])
        time_qual = dict()
        for i in filenames:
            lines = []
            r = i +"_"+alg+"_600_(.*).trace"
            for j in os.listdir("output/"+alg+"/"):
                matchObj = re.match(r, j)
                if matchObj:
                    with open('output/'+alg+'/'+j, 'r') as f:
                        for line in f.readlines():
                            time,qual = line.strip().split(',')
                        lines.append((float(time), float(qual)))
            time_qual[i] = tuple(lines)

        for i in time_qual:
            # print i, time_qual[i]
            a = np.array(time_qual[i])
            b = np.average(a, axis=0)
            print i
            print b