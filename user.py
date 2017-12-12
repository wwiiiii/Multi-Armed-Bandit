import numpy as np

class User:
	def __init__(self, posProb, itemProb):
		self.L = len(posProb)
		self.K = len(itemProb)
		self.posProb = posProb
		self.itemProb = itemProb

	def react(self, items):
		assert len(items) == self.L
		clicked = [False for _ in range(self.L)]
		for i in range(self.L):
			if 0 == np.random.binomial(1, self.posProb[i]): continue
			if 0 == np.random.binomial(1, self.itemProb[items[i]]): continue
			clicked[i] = True
		return clicked