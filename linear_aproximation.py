# the usual imports
import pandas as pd
import numpy as np
from scipy import spatial
# importing the network environment class
from environment import network
# to scale the feature vectors
from sklearn.preprocessing import StandardScaler

class Model:
	def __init__(self):
		self.theta = np.random.randn(126*127) / 2 # initializes the weights
		self.scaler = StandardScaler()
		self.scaler.mean_ = np.loadtxt('mean_.csv', delimiter=',')
		self.scaler.scale_ = np.loadtxt('scale_.csv', delimiter=',')

	# the naive feature construction
	@staticmethod
	def sa2x(state_vector, action):
		# converts the firsts 21 elements (azimuth) to radians
		x1 = np.radians(state_vector.astype(float)[:21])
		x2 = (state_vector[21:42] * state_vector[42:64])
		# one-hot encodes the 126 posible actions
		x3 = np.zeros(126)
		x3[action] = 1
		# concatenates the vectors
		x = np.append(x1, x2)
		x = np.append(x, x3)
		# adds the bias and returns
		return np.append(x, 1)

	@staticmethod
	def intercell_distance(state_vector, cell_index):
		thershold = np.cos(np.radians(60)) # sites to be considered are -60/+60 degrees around
		L = 100 # cell_vector length
		n_cells = 21
		cell_on_starts = 84
		lon_starts_at = 21
		lat_starts_at = 0
		azimuth_starts_at = 42
		# first to check if the cell is on
		if state_vector[cell_on_starts + cell_index]:
			x0 = state_vector[lon_starts_at + cell_index]
			y0 = state_vector[lat_starts_at + cell_index]
			# generates the distance vector for all the other sites respect to the cell
			dist_vector = np.array([((state_vector[lon_starts_at + i] - x0),
									(state_vector[lat_starts_at + i] - y0))
									for i in np.arange(n_cells)])
			dist_vector = np.unique(dist_vector, axis = 0)
			# removes the [0,0] from the array
			dist_vector = [dist for dist in dist_vector if dist.dot(dist) != 0]
			# generate the vector for the cell
			azimuth_rad = L * np.radians(state_vector[azimuth_starts_at + cell_index])
			cell_vector = L * np.array([np.cos(azimuth_rad), np.sin(azimuth_rad)])
			# calculates the cosine similarity between the cell_vector and the site vectors
			cossim_vector = [1 - spatial.distance.cosine(cell_vector, site)
							for site in dist_vector]
			index_max_n = np.where(cossim_vector > thershold)[0]
			if len(index_max_n) > 0:
				n_sites = [dist_vector[i] for i in index_max_n]
				dist = [np.sqrt(n_sites[i].dot(n_sites[i])) for i in np.arange(len(n_sites))]
				min_dist = np.min(dist)
			# end if
			else:
				min_dist = 2000
		else:

			min_dist = 1
		#end else
		#end if
		return min_dist

	# the feature construction version 1 10/7/2017
	def s2x_v1(self, s):
		cell_on_starts = 84
		lon_starts_at = 21
		lat_starts_at = 0
		azimuth_starts_at = 42
		txpower_starts_at = 63
		# converts the azimuth to radians
		x1 = np.radians(s.astype(float)[azimuth_starts_at:azimuth_starts_at+21])
		# cell-i.txpower*cell_on
		x2 = s[txpower_starts_at:txpower_starts_at+21] * s[cell_on_starts:cell_on_starts+21]
		# cell-i.txpower*cell_on*sin(azimuth)
		x3 = x2 * np.sin(x1)
		# cell-i.txpower*cell_on*cos(azimuth)
		x4 = x2 * np.cos(x1)
		# cell-i.(txpower*cell_on) ^ 2
		x5 = x2 * x2
		# ( 1 / intercell_distance) ^ 2
		x6 = [ ( 100 / self.intercell_distance(s, i)) for i in np.arange(21) ]
		# concatenates the vectors
		x = np.append(x1, x2) # len(x) = 42
		x = np.append(x, x3) # len 63
		x = np.append(x, x4) # len 84
		x = np.append(x, x5) # len 105
		x = np.append(x, x6) # len 126
		# adds the bias
		x = np.append(x, 1) # len 127
		x_trans = self.scaler.transform(x.reshape(1,-1))
		# returns the scaled version
		return x_trans[0]

	# the feature construction version 1 10/7/2017, including action
	def sa2x_v1(self, s, a):
		a = int(a)
		cell_on_starts = 84
		lon_starts_at = 21
		lat_starts_at = 0
		azimuth_starts_at = 42
		txpower_starts_at = 63
		# converts the azimuth to radians
		x1 = np.radians(s.astype(float)[azimuth_starts_at:azimuth_starts_at+21])
		# cell-i.txpower*cell_on
		x2 = s[txpower_starts_at:txpower_starts_at+21] * s[cell_on_starts:cell_on_starts+21]
		# cell-i.txpower*cell_on*sin(azimuth)
		x3 = x2 * np.sin(x1)
		# cell-i.txpower*cell_on*cos(azimuth)
		x4 = x2 * np.cos(x1)
		# cell-i.(txpower*cell_on) ^ 2
		x5 = x2 * x2
		# ( 1 / intercell_distance) ^ 2
		x6 = [ ( 100 / self.intercell_distance(s, i)) for i in np.arange(21) ]
		# concatenates the vectors
		x = np.append(x1, x2) # len(x) = 42
		x = np.append(x, x3) # len 63
		x = np.append(x, x4) # len 84
		x = np.append(x, x5) # len 105
		x = np.append(x, x6) # len 126
		# adds the bias and returns
		x = np.append(x, 1) # len 127
		# transforms the feature vector
		x_trans = self.scaler.transform(x.reshape(1,-1))[0]
		# creates the final x vector, 126 actions times 127 features
		final_x = np.zeros(126 * 127)
		initial_index = a * 127
		final_index = (a + 1) * 127
		final_x[initial_index:final_index] = x_trans
		return final_x

	def hat(self, s, a):
		x = self.sa2x_v1(s, a)
		return self.theta.dot(x)

	def grad(self, s, a):
		# since it is a linear aproximation, the gradient is the same feature vector
		return self.sa2x_v1(s, a)

	def getQs(self, s):
		Qs = {}
		for a in range(126):
			q_sa = self.hat(s, a)
			Qs[a] = q_sa
		return Qs

class e_greedy_timedecay:
	def __init__(self, action_space, initial_epsilon = 0.8, decaying_factor = 0.9999):
		self.epsilon = initial_epsilon
		self.decaying_factor = decaying_factor
		self.action_space = action_space

	def decay(self):
		self.epsilon *= self.decaying_factor

	def get_action(self, model, state_vector):
		# generates a random number
		rd = np.random.rand(1)[0]
		print('rd: {}, eps: {}'.format(rd, self.epsilon))
		if rd >= self.epsilon: # if will exploit
			print('exploit..')
			Qs = model.getQs(state_vector)
			# print(Qs)
			action = max(Qs, key=Qs.get)
			# print('action: {}'.format(action))
		else: # it will explore
			print('explore..')
			action = np.random.choice(self.action_space)
			# print('action: {}'.format(action))
		return action

