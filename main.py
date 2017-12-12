import os
import time
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

from simulator import Simulator

def main():
	'''K = 25# number of items, 0-base
	L = 10# number of position, 0-base
	itemid = [i for i in range(K)]
	posProb = [0.5] * L # observation probability for certain position, decreasing order
	itemProb = [0.05] * K # click probability when observed for certain item
	'''
	if not os.path.exists('log'):
		os.makedirs('log')

	'''K = 7
	L = 3
	itemid = [i for i in range(K)]
	posProb = [0.8, 0.5, 0.3]
	itemProb = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]'''

	start_time = time.time()
	posProb = [1.0]
	itemProb = [0.45, 0.44, 0.40, 0.25, 0.05]
	K = len(itemProb)
	L = len(posProb)
	itemid = [i for i in range(K)]

	expCount = 50
	stepCount = 5000
	sim = Simulator(itemid, posProb, itemProb)
	log_fname='log/'+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.')

	'''
	base, plots = sim.run(step_cnt=stepCount, log_fname=log_fname)
	for i in range(expCount-1):
		print(str(i+2)+'th exp')
		base, nowp = sim.run(step_cnt=stepCount, log_fname=log_fname)
		plots += nowp
	'''

	with multiprocessing.Pool(10) as p:
		exp_result = p.map(sim.run, [
				{'step_cnt':stepCount, 'log_fname':log_fname} for _ in range(expCount)
			])
	base = exp_result[0][0]
	plots = sum([i for _, i in exp_result])
	plots = plots / float(expCount)

	labels = ['TS', 'BayesUCB', 'UCB1', 'UCBtune', 'Random']
	plt.figure()
	for i in range(len(labels)):
		plt.plot(base, list(plots[i]), label=labels[i])
	plt.legend()
	plt.xlabel('step')
	plt.ylabel('regret')
	plt.savefig(log_fname+'graph.png')

	plt.figure()
	for i in range(len(labels)-1):
		plt.plot(base, list(plots[i]), label=labels[i])
	plt.legend()
	plt.xlabel('step')
	plt.ylabel('regret')
	plt.savefig(log_fname+'graph_without_random.png')

	with open(log_fname+'txt', 'w') as f:
		f.write(str(('labels', labels, '# of arms', K, '# of pos', L, 'posProb', posProb, 'itemProb', itemProb, 'expCnt', expCount, 'stepCnt', stepCount)))

	with open(log_fname+'.plot.pickle', 'wb') as f:
		pickle.dump(plots, f)

	print('%s seconds' % (time.time() - start_time))

if __name__ == '__main__':
	main()
