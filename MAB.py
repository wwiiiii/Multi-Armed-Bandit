import numpy as np
import scipy as sp
import random
import pickle

class MAB:
	def __init__(self, itemid, posProb):
		self.K = len(itemid)
		self.L = len(posProb)
		self.turn = 0 # how many times item list selected
		self.itemid = itemid
		self.posProb = posProb

	#return sorted list of item numbers, with length require_num. Default Random
	def select_items(self, required_num):
		random.shuffle(self.itemid)
		return self.itemid[:required_num]

	def update(self, selected_items, feedback):
		pass
	
	def save(self, fname):
		with open(fname, 'wb') as f:
			pickle.dump(self.__dict__, f)

	def load(self, fname):
		with open(fname, 'rb') as f:
			sim = pickle.load(f)
			self.__dict__.clear()
			self.__dict__.update(sim)