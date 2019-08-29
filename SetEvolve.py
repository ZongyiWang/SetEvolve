import sys
from BaseGraphicalLasso import BaseGraphicalLasso
from StaticGL import StaticGL
import numpy as np
import time
from DataHandler import DataHandler
import random
from scipy.stats import norm
import math
from numpy.linalg import inv
import penalty_functions as pf
import multiprocessing
import traceback


def mp_static_gl((theta, z0, u0, emp_cov_mat, rho,
                  lambd, eta, dimension, max_iter)):

    # Multiprocessing worker computing the
    # Static Graphical Lasso for given subset
    # of blocks

    try:
        iteration = 0
        stopping_criteria = False
        theta_pre = []
        while iteration < max_iter and stopping_criteria is False:
            """ Theta update """
            a = z0 - u0
            at = a.transpose()
            m = eta*(a + at)/2 - emp_cov_mat
            d, q = np.linalg.eig(m)
            qt = q.transpose()
            sqrt_matrix = np.sqrt(d**2 + 4/eta*np.ones(dimension))
            diagonal = np.diag(d) + np.diag(sqrt_matrix)
            theta = np.real(
                eta/2*np.dot(np.dot(q, diagonal), qt))
            """ Z-update """
            z0 = pf.soft_threshold_odd(theta + u0, lambd, rho)
            """ U-update """
            u0 += theta - z0
            """ Check stopping criteria """
            if iteration > 0:
                dif = theta - theta_pre
                fro_norm = np.linalg.norm(dif)
                if fro_norm < 1e-5:
                    stopping_criteria = True
            theta_pre = list(theta)
            iteration += 1
    except Exception as e:
        traceback.print_exc()
        raise e
    return theta


class SetEvolve(StaticGL):

    """ Initialize attributes, read data """
    def __init__(self, *args, **kwargs):
        super(SetEvolve, self).__init__(processes=10,
                                       *args, **kwargs)

        # self.eta = float(self.obs)/float(3*self.rho)
        # self.eta = float(self.obs)/float(self.rho)

    """ Assigns rho based on number of observations in a block """
    def get_rho(self):
        return super(SetEvolve,self).get_rho()
        # return float(self.obs + 0.1) / float(3)

    """ Main modifications """
    def stepone(self,Y):
        transform = np.array(Y)
        transform = self.gaussCopula(transform)
        return transform


    def gaussCopula(self,Y):
        for i in range(len(Y)):
            mean = np.mean(Y[i])
            std = np.std(Y[i])
            if self.penalty_function == 2:
                Y[i] = mean + std * self.hfunc(Y[i], mean, std)
            else:
                Y[i] = mean + std * self.countDiscrete(Y[i])
        return Y

    def countDiscrete(self, vec):
        output = []
        sort = sorted(vec)
        n = len(vec)
        delta = 1.0 / (4 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))
        mapping = {}
        for j in range(1, n):
            if sort[j] != sort[j - 1]:
                mapping[sort[j - 1]] = (j - 1.0)/n
        mapping[sort[j]] = 1.0
        for j in range(n):
            if mapping[vec[j]] < delta:
                output.append(norm.ppf(delta))
            elif mapping[vec[j]] > 1 - delta:
                output.append(norm.ppf(1 - delta))
            else:
                output.append(norm.ppf(mapping[vec[j]]))
        return np.array(output)

    def hfunc(self,vec, mean, std):
        output = []
        n = len(vec)
        delta = 1.0 / (4 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))
        for i in range(n):
            zscore = (vec[i] - mean)/std
            percent = norm.cdf(zscore)
            if percent < delta:
                output.append(norm.ppf(delta))
            elif percent > 1 - delta:
                output.append(norm.ppf(1 - delta))
            else:
                output.append(norm.ppf(percent))
        return np.array(output)


    def heat_kernel(self,t,x,y,d):
        if not t:
            return 1.0 
        else:
            return 1.0/(4.0 * np.pi * t) ** (d/2.0) * np.exp(-1.0 * abs(x- y)**2/(4.0 * t))


    """ Read data from the given file. Get parameters of data
        (number of data samples, observations in a block).
        Compute empirical covariance matrices.
        Compute real inverse covariance matrices,
        if provided in the second line of the data file. """

    def read_data(self, filename, comment="#", splitter=","):
        with open(filename, "r") as f:
            comment_count = 0
            for i, line in enumerate(f):
                if comment in line:
                    comment_count += 1
                else:
                    if self.dimension is None:
                        if self.datecolumn:
                            self.dimension = len(line.split(splitter)) - 1
                        else:
                            self.dimension = len(line.split(splitter))
        self.datasamples = i + 1 - comment_count
        self.obs = self.datasamples / self.blocks
        all_lst = []
        with open(filename, "r") as f:
            count = 0
            for i, line in enumerate(f):
                if comment in line:
                    if i == 0 and 'case_study' in filename:
                        self.entity_list = line.rstrip().split(',')[1:]
                    if i == 1:
                        self.generate_real_thetas(line, splitter)
                    continue
                if count == 0 and self.datecolumn is True:
                    start_date = line.strip().split(splitter)[0]
                if self.datecolumn:
                    all_lst.append([float(x)
                                for x in np.array(line.strip().
                                                  split(splitter)[1:])])
                else:
                    all_lst.append([float(x)
                                for x in np.array(line.strip().
                                                  split(splitter))])
                count += 1
        all_lst = np.array(all_lst)
        all_lst = self.stepone(all_lst)



        with open(filename, "r") as f:
            # lst = []
            block = 0
            count = 0
            totalcount = 0
            for i, line in enumerate(f):
                if comment in line:
                    if i == 1:
                        self.generate_real_thetas(line, splitter)
                    continue
                if count == 0 and self.datecolumn is True:
                    start_date = line.strip().split(splitter)[0]
                count += 1
                totalcount += 1
                if count == self.obs:
                    if self.datecolumn:
                        end_date = line.strip().split(splitter)[0]
                        self.blockdates[block] = start_date + " - " + end_date

                    self.emp_cov_mat[block] = np.zeros((self.dimension,self.dimension))
                    for i in range(self.obs):
                        coeff = [self.heat_kernel(abs(j - (totalcount - i)),0,0,self.dimension) for j in range(len(all_lst))]
                        weightsum = sum(coeff)
                        coeff_matrix = np.diag(coeff/weightsum)
                        self.emp_cov_mat[block] += np.dot(np.dot(all_lst.transpose(), coeff_matrix),all_lst)
                        # for j in range(len(all_lst)):
                            # self.emp_cov_mat[block] += np.outer(all_lst[j],all_lst[j].transpose()) * self.heat_kernel(abs(j - (totalcount - i)),0,0,self.dimension)/weightsum
                    self.emp_cov_mat[block] = np.real(
                        self.emp_cov_mat[block]/self.obs)

                    # lst = []
                    count = 0
                    block += 1

    def run_algorithm(self, max_iter=10000):
        start_time = time.time()
        p = multiprocessing.Pool(self.processes)
        inputs = [(self.thetas[i], self.z0s[i], self.u0s[i],
                   self.emp_cov_mat[i], self.rho,
                   self.lambd, self.eta, self.dimension, max_iter)
                  for i in range(self.blocks)]
        self.thetas = p.map(mp_static_gl, inputs)
        p.close()
        p.join()
        self.run_time = '{0:.3g}'.format(time.time() - start_time)
        self.thetas = [np.round(theta, self.roundup) for theta in self.thetas]


if __name__ == "__main__":
    if not (len(sys.argv) == 5 or len(sys.argv) == 6):
        print("input number error!")
        print("4 or 5 inputs required")
        print("1.filename")
        print("2.discrete param(discrete=2, otherwise=1)")
        print("3.number of blocks")
        print("4.lambda")
        print("5.output result matrix\n")
        sys.exit()

    print("\nSetEvolve")
    # Input parameters from command line:
    #  1. Data file in csv format
    #  2. lambda

    start_time = time.time()
    datahandler = DataHandler()

    """ Parameters for creating solver instance """
    filename = sys.argv[1]
    real_data = True
    if "synthetic_data" in filename:
        real_data = False
    chosenf = int(sys.argv[2])
    block = int(sys.argv[3])
    lambd = float(sys.argv[4])
    draw = 0
    if len(sys.argv) == 6:
        draw = int(sys.argv[5])


    """ Create solver instance """
    # print "\nReading file: %s\n" % filename
    solver = SetEvolve(filename=filename,lambd=lambd,datecolumn=real_data,blocks=block, penalty_function=chosenf)

    print "Total data samples: %s" % solver.datasamples
    print "Blocks: %s" % solver.blocks
    print "Observations in a block: %s" % solver.obs
    print "Rho: %s" % solver.rho
    print "Lambda: %s" % solver.lambd
    print "Beta: %s" % solver.beta
    print "Penalty function: %s" % solver.penalty_function
    print "Processes: %s" % solver.processes

    """ Run algorithm """
    # print "\nRunning algorithm..."
    solver.run_algorithm()

    """ Evaluate and print results """
    for i in range(solver.blocks):
        solver.print_matrix(i)
    solver.get_ready_draw(draw)
    
    print "\nTemporal deviations: "
    solver.temporal_deviations()
    print solver.deviations
    print "Normalized Temporal deviations: "
    print solver.norm_deviations
    try:
        print "Temp deviations ratio: {0:.3g}".format(solver.dev_ratio)
    except ValueError:
        print "Temp deviations ratio: n/a"

    """ Evaluate and create result file """
    if not real_data:
        solver.correct_edges()
        print "\nAverage number of Edges predicted: %s" % str(solver.all_positives*1.0/solver.blocks)
        print "\nTotal Edges: %s" % solver.real_edges
        print "Correct Edges: %s" % solver.correct_positives
        print "Total Zeros: %s" % solver.real_edgeless
        false_edges = solver.all_positives - solver.correct_positives
        print "False Edges: %s" % false_edges
        print "MacroF1 Score: %s" % solver.f1score
        print "MicroF1 Score: %s" % solver.f1micro
        # datahandler.write_results(filename, solver)
    else:
        datahandler.write_network_results(filename, solver)

    """ Running times """
    print "\nAlgorithm run time: %s seconds" % (solver.run_time)
    print "Execution time: %s seconds" % (time.time() - start_time)
