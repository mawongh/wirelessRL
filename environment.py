import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

class network():
	# filenames
	default_remfile = 'lena-simple_env.rem'
	tempfile = 'lena-simple_env_temp.csv'

	def __init__(self, output_filename = default_remfile):
		self.output_filename = output_filename
		# environment size
		self.x_min = 300
		self.y_min = 0
		self.x_max = 3300
		self.y_max = 5000

	def readfromcsv(self, input_filename = 'simple_env_conf.csv'):
		self.conf = pd.read_csv(input_filename)
		self.check_conf()

	def writetocsv(self, conf_filename):
		self.conf.to_csv(conf_filename, index = False)

	def loadrem(self):
		self.rem = pd.read_csv(self.output_filename, sep='\t', header=None)
		# adding a fifth column as the SINR in dB
		self.rem.loc[:,4] = 10*np.log(self.rem.loc[:,3])
		self.rem.columns = ['x', 'y', 'z', 'SINR', 'SINR_dB']
		self.rem['point'] = [(x, y) for x,y in zip(self.rem['x'], self.rem['y'])]

	def runsimulation(self):
		# write the configuration into a csv file
		print('Storing configuration...')
		self.writetocsv(network.tempfile)
		# running the simulation
		print('Running the simulation...')
		subprocess.call("cd ~/ws/bake/source/ns-3.26;./waf --run scratch/lena-simple_env",\
						shell=True)
		# removing the temp file
		print('Removing temp file...')
		shellcommand = 'rm ~/ws/bake/source/ns-3.26/scratch/' + network.tempfile
		subprocess.call(shellcommand, shell=True)
		# updating the rem
		self.loadrem()

	def plotrem(self):
		fig = plt.figure(figsize=(7,8))
		ax = fig.add_subplot(111)
		ax.scatter(self.rem['x'], self.rem['y'], s=25,\
				c = self.rem['SINR_dB'], marker='s', edgecolors = 'none')
		# plt.colorbar()
		ax.set_xlim(self.x_min, self.x_max)
		ax.set_ylim(self.y_min, self.y_max)
		plt.show()

	def cdfrem(self):
		pass

	def gen_ue_dist(self, N):
		# proportion of the total users per random distribution
		ue_prop = {1:0.75, 2:0.05, 3:0.05, 4:0.05, 5:0.05, 6:0.05}
		# centers and covariance matrixes for the normal distributions
		center_1 = [1500, 2750]
		sd_1 = [[150000, 0], [0, 150000]]
		center_2 = [1200, 800]
		sd_2 = [[50000, 0], [0, 200000]]
		center_3 = [600, 4300]
		sd_3 = [[20000, 0], [0, 100000]]
		center_4 = [2500, 2000]
		sd_4 = [[300000, 0], [0, 20000]]
		center_5 = [2800, 3900]
		sd_5 = [[100000, 0], [0, 100000]]
		# generates the ue distribution based on five (5) multivariate normal dist
		# and one (1) uniform distribution

		def closest_point(point, points):
			""" Find closest point from a list of points. """
			return points[cdist([point], points).argmin()]

		x1,y1 = np.random.multivariate_normal(center_1, sd_1, np.int(N * ue_prop[1])).T
		self.ue_dist = pd.DataFrame({'x':x1, 'y':y1})
		x2, y2 = np.random.multivariate_normal(center_2, sd_2, np.int(N * ue_prop[2])).T
		self.ue_dist = pd.concat([self.ue_dist, pd.DataFrame({'x':x2, 'y':y2})], ignore_index=True)
		x3, y3 = np.random.multivariate_normal(center_3, sd_3, np.int(N * ue_prop[3])).T
		self.ue_dist = pd.concat([self.ue_dist, pd.DataFrame({'x':x3, 'y':y3})], ignore_index=True)
		x4, y4 = np.random.multivariate_normal(center_4, sd_4, np.int(N * ue_prop[4])).T
		self.ue_dist = pd.concat([self.ue_dist, pd.DataFrame({'x':x4, 'y':y4})], ignore_index=True)
		x5, y5 = np.random.multivariate_normal(center_5, sd_5, np.int(N * ue_prop[5])).T
		self.ue_dist = pd.concat([self.ue_dist, pd.DataFrame({'x':x5, 'y':y5})], ignore_index=True)
		x6 = np.random.uniform(self.x_min, self.x_max, np.int(N * ue_prop[6]))
		y6 = np.random.uniform(self.y_min, self.y_max, np.int(N * ue_prop[6]))
		self.ue_dist = pd.concat([self.ue_dist, pd.DataFrame({'x':x6, 'y':y6})], ignore_index=True)
		# filters out the UEs falling outside the environment box
		self.ue_dist = self.ue_dist[(self.ue_dist.x >= self.x_min ) & (self.ue_dist.x <= self.x_max)\
						   & (self.ue_dist.y <= self.y_max) & (self.ue_dist.x >= self.y_min)]
		self.ue_dist['point'] = [(x, y) for x,y in zip(self.ue_dist['x'], self.ue_dist['y'])]
		self.ue_dist['closest'] = [closest_point(x, list(self.rem['point'])) for x in self.ue_dist['point']]


	def plot_ue_dist(self):
		fig = plt.figure(figsize=(7,8))
		ax = fig.add_subplot(111)
		ax.plot(self.ue_dist.x, self.ue_dist.y, 'x')
		ax.set_xlim(self.x_min, self.x_max)
		ax.set_ylim(self.y_min, self.y_max)
		plt.show()

	def reward(self, variant = 2):

		thershold_percentile = 1 - 0.95
		thershold_SINR = 10
		# defining the functions to determine the closest points and lookups

		def match_value(df, col1, x, col2):
			""" Match value x from col1 row to value in col2. """
			return df[df[col1] == x][col2].values[0]

		self.ue_dist['SINR_dB'] = [match_value(self.rem, 'point', x, 'SINR_dB') for x in self.ue_dist['closest']]

		# returning the reward according to the requested variant
		if variant == 1:
			return (self.ue_dist['SINR_dB'].quantile(thershold_percentile) >= thershold_SINR) * 100
		if variant == 2:
			return self.ue_dist['SINR_dB'].quantile(thershold_percentile)


	def getstate_vector(self):
		vector = np.array(self.conf['azimuth'].tolist() + \
						  self.conf['txpower'].tolist() + \
						  self.conf['cell_on'].tolist())
		return vector

	def getstate_vector2(self):
		vector = np.array(self.conf['lat'].tolist() + self.conf['lon'].tolist() + \
			self.conf['azimuth'].tolist() + self.conf['txpower'].tolist() + \
			self.conf['cell_on'].tolist())
		return vector

	@staticmethod
	def action_code_description(action_code):
		# dictionary of cell action
		attribute_dict = {0:'azimuth', 1:'txpower', 2:'cell_on'}
		action_dict = {0:'incr20', 1:'decr20', 2:'incr2', 3:'decr2', 4:'on', 5:'off'}

		cell = int(action_code / 6) # this operations gives the cell on the action will be executed
		subaction = action_code % 6
		attribute = int(subaction / 2)

		# initialising the description String
		descr = 'cell_{}.{}.{}'.format(cell, attribute_dict[attribute], action_dict[subaction])
		print(action_code, descr)
		# return descr
		return (cell, attribute_dict[attribute], action_dict[subaction])

	def execute_action(self, action_code):
		# get the action required for the action code
		cell, attribute, action = self.action_code_description(action_code)
		if action == 'incr20':
			self.conf.loc[cell, attribute] += 20
		if action == 'decr20':
			self.conf.loc[cell, attribute] -= 20
		if action == 'incr2':
			self.conf.loc[cell, attribute] += 2
		if action == 'decr2':
			self.conf.loc[cell, attribute] -= 2
		if action == 'on':
			self.conf.loc[cell, attribute] = 1
		if action == 'off':
			self.conf.loc[cell, attribute] = 0
		# checks and corrects the configuration
		self.check_conf()
		# runs the simulation
		self.runsimulation()
		# generates a new ue distribution (this makes the reward stochastic), comment for determinist
		self.gen_ue_dist(300) # 300 users
		# return a tuple with the new state vector and the reward
		return (self.getstate_vector(), self.reward())

	def check_conf(self):
		"""checks and corrects if the configuration is valid"""
		cell0s = [(i%3==0) for i in range(21)]
		cell1s = [(i%3==1) for i in range(21)]
		cell2s = [(i%3==2) for i in range(21)]
		## the azimuth values for first cells in each site sould be between 0 and 100
		self.conf.azimuth[cell0s & (self.conf.azimuth < 0)] = 0
		self.conf.azimuth[cell0s & (self.conf.azimuth > 100)] = 100
		## the azimuth values for second cells in each site sould be between 120 and 220
		self.conf.azimuth[cell1s & (self.conf.azimuth < 120)] = 120
		self.conf.azimuth[cell1s & (self.conf.azimuth > 220)] = 220
		## the azimuth values for second cells in each site sould be between 240 and 340
		self.conf.azimuth[cell2s & (self.conf.azimuth < 240)] = 240
		self.conf.azimuth[cell2s & (self.conf.azimuth > 340)] = 340
		## the txpower for all cells should be between 36 and 46 dBm
		self.conf.txpower[self.conf.txpower < 36] = 36
		self.conf.txpower[self.conf.txpower > 46] = 46

	def reset(self):
		""" will set the state of to a random state"""
		# azimuths
		self.conf.azimuth = np.tile([0, 120, 240], 7) + \
							np.random.randint(6, size = 21) * 20
		# txpower
		self.conf.txpower = 36 + np.random.randint(6, size = 21) * 2
		# cell_on
		self.conf.cell_on = np.random.randint(2, size = 21)




def test_network_class():
	conf_file = 'simple_env_conf.csv'
	
	print('creating the class...')
	net1 = network()

	print('reading the class from file...')
	net1.readfromcsv(conf_file)

	# print('running the simulation...\n')
	# net1.runsimulation()

	print('loading rem file...')
	net1.loadrem()

	print('plot the output remfile...')
	net1.plotrem()

if __name__ == '__main__':
	# runs the test_network_class function
	test_network_class()
