import numpy as np
import random
from scipy.stats import norm
import math
from numpy.linalg import inv

# input 
p = 20 # variable size
# n = 500 # sample size 
edge_size = 20 # how many edges are connnected 
T = 10 # timestamp 
count = str(T)
N = 10 # max observation count 
T_0 = 5 # timestamps wait to evolve
have_noise = False
lamb = 1	# poisson noise parameter
per_timestamp = 100

def add_noise(vec,n):
	get = np.random.poisson(0.5, n)
	return vec + get


## generate the first precision matrix 
x_axis = random.sample(range(p),edge_size)
y_axis = random.sample(range(p),edge_size)
mean = [0]*p
inv_cov = [[0] * p for i in range(p)]
for idx in range(edge_size):
	rand = np.random.uniform(0.1,0.3)
	inv_cov[x_axis[idx]][y_axis[idx]] = rand
	inv_cov[y_axis[idx]][x_axis[idx]] = rand
	inv_cov[idx][idx] = 3 #change the diagnoal element to make matrix positive definite

eigenvalues = np.linalg.eigvals(inv_cov) 

# generate the underlying multivariate Gaussian data
Z = np.random.multivariate_normal(mean, inv(inv_cov))
# generate observed counts based on Z
Y = np.array([math.floor(norm.cdf(Z[i],0,inv(inv_cov)[i][i])*N) for i in range(len(Z))])



# time varying part 
for t in range(0, T):
	if t % T_0 == 0:
		# in every change, we remove 5 existing edges, and add 5 new edges
		for i in range(5):
			if x_axis[i] != y_axis[i]:
				inv_cov[x_axis[i]][y_axis[i]] = 0
				inv_cov[y_axis[i]][x_axis[i]] = 0
		x_axis_new = random.sample(range(p),5)
		y_axis_new = random.sample(range(p),5)
		add_pool = np.random.uniform(0.1,0.3,5)
		x_axis = x_axis[5:] + x_axis_new 
		y_axis = y_axis[5:] + y_axis_new
		dec_pool = []
		for i in range(5):
			if x_axis[i] != y_axis[i]:
				dec_pool.append(inv_cov[x_axis[i]][y_axis[i]])
			else:
				dec_pool.append(0)
				
	if t % T_0 == int(T_0/2):
		with open('networks/our_network' + count + '_' + str(int(t / T_0)) + '.csv','w') as wf:
			wf.write('# Unidirectional network\n')
			wf.write('# Weight for edge between same nodes is 1 by default\n')
			wf.write('# First node,Second node,Weight\n')
			for i in range(p):
				for j in range(p):
					if i < j and inv_cov[i][j] > 0:
						wf.write(",".join([str(i+1), str(j+1), str(inv_cov[i][j])]) + '\n')

		with open('networks/our_network' + count + '_' + str(int(t / T_0)) + '_matrix.csv','w') as wf:
			for i in range(p):
				# temp = inv_cov[i][:]
				# temp[i] = 1.0
				# wf.write(",".join(map(str,temp)) + '\n')		
				# print(temp)
				# print(inv_cov[i])
				wf.write(",".join(map(str,inv_cov[i])) + '\n')		

	for i in range(5):
		inv_cov[x_axis_new[i]][y_axis_new[i]] += add_pool[i]/float(T_0)
		inv_cov[y_axis_new[i]][x_axis_new[i]] += add_pool[i]/float(T_0)
		if x_axis[i] != y_axis[i]:
			inv_cov[x_axis[i]][y_axis[i]] -= dec_pool[i]/float(T_0)
			inv_cov[y_axis[i]][x_axis[i]] -= dec_pool[i]/float(T_0)


	# generate the underlying multivariate Gaussian data
	for i in range(per_timestamp):
		Z = np.random.multivariate_normal(mean, inv(inv_cov))
		# generate observed counts based on Z
		Y_cur = np.array([math.ceil(norm.cdf(Z[i],0,inv(inv_cov)[i][i])*N) for i in range(len(Z))])
		if have_noise:
			Y_cur = add_noise(Y_cur, len(Y_cur))
		Y = np.vstack((Y, Y_cur))


wf = None
if have_noise:
	wf = open('synthetic_data/our_synthetic' + count + '_noised.csv','w')
else:
	wf = open('synthetic_data/our_synthetic' + count + '.csv','w')
wf.write('# Data generated from networks:\n')
wf.write('# ')
for i in range(int(T/T_0) - 1):
	wf.write('networks/our_network' + count + '_' + str(i) + '.csv: ' + str(T_0*per_timestamp) + ', ')
wf.write('networks/our_network' + count + '_' + str(int(T/T_0) - 1) + '.csv: ' + str(T_0*per_timestamp) + '\n')

Y = Y[1:]
for i in range(T * per_timestamp):
	wf.write(",".join(map(str,list(Y[i]))) + '\n')


