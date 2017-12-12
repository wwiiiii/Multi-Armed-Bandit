import numpy as np
import scipy as sp
import pickle
from scipy.stats import beta
from MAB import MAB

class BayesUCB(MAB):
	def __init__(self, itemid, posProb):
		super().__init__(itemid, posProb)
		self.S = [[1 for _ in range(self.L)] for __ in range(self.K)]
		self.N = [[2 for _ in range(self.L)] for __ in range(self.K)]
	
	#return sorted list of item numbers, with length require_num
	def select_items(self, required_num):
		self.turn += 1
		sample_val = [0.0] * self.K
		for k in range(self.K):
			z0 = beta.ppf(1.0 - 1.0 / self.turn, sum(self.S[k]), sum(self.N[k]) - sum(self.S[k]))
			sample_val[k] = z0
		items = sorted([(sample_val[k], k) for k in range(self.K)], reverse=True)
		result = [items[i][1] for i in range(required_num)]
		return result

	def update(self, selected_items, feedback):
		assert len(selected_items) == len(feedback)
		for l in range(len(selected_items)):
			k = selected_items[l]
			self.N[k][l] += 1
			if feedback[l]:
				self.S[k][l] += 1