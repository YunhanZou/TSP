"""
Author: Xia Wu
CSE 6140, Fall 2018, Georgia Tech

Parse Output solution and trace files
"""

class Output(object):
    def __init__(self, filename, method, cutoff, algo, rand_seed=0):
        dir, filename = filename.split("/")
        inst, suffix = filename.split(".")
        # print inst, method, cutoff
        self.path = 'output/' + str(algo) + '/' + str(inst) + '_' + str(method) + '_' + str(int(cutoff))
        if rand_seed:
            self.path += '_' + str(rand_seed)

    def solution(self, sol_list):
        path = self.path
        path += '.sol'
        f = open(path, 'w')
        line1 = str(int(sol_list[0])) + '\n'
        f.write(line1)

        vertices = sol_list[1:]
        line2 = ','.join(str(v + 1) for v in vertices)
        f.write(line2)

        f.close()

    def sol_trace(self, sol_trace_list):
        path = self.path
        path += '.trace'
        f = open(path, 'w')
        for trace in sol_trace_list:
            line = str(trace[0]) + ', ' + str(int(trace[1])) + '\n'
            f.write(line)
        f.close()


if __name__ == '__main__':
    test1 = [100, 1, 2, 3, 4, 5, 6, 1]
    o = Output('test1', 'LS1', 600, 4)
    o.solution(test1)

    test2 = [20000, 10, 20, 30, 40, 60, 10]
    o = Output('test2', 'Approx', 600)
    o.solution(test2)

    test3 = [(1.2, 123), (2.3, 234), (3.4, 345), (4.5, 456)]
    o = Output('test3', 'BnB', 600)
    o.sol_trace(test3)

