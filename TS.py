import numpy as np
import scipy as sp
import pickle
from scipy.stats import beta
from MAB import MAB

class TS(MAB):
	def __init__(self, itemid, posProb):
		super().__init__(itemid, posProb)
		self.S = [0 for _ in range(self.K)]
		self.N = [0 for _ in range(self.K)]

	#return sorted list of item numbers, with length require_num
	def select_items(self, required_num):
		self.turn += 1
		sample_val = [0.0] * self.K
		for k in range(self.K):
			sample_val[k] = np.random.beta(self.S[k] + 1, self.N[k] - self.S[k] + 1)
		items = sorted([(sample_val[k], k) for k in range(self.K)], reverse=True)
		result = [items[i][1] for i in range(required_num)]
		return result

	def update(self, selected_items, feedback):
		assert len(selected_items) == len(feedback)
		for l in range(len(selected_items)):
			k = selected_items[l]
			self.N[k] += 1
			if feedback[l]:
				self.S[k] += 1