import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt

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

	def readfromcsv(self, input_filename):
		self.conf = pd.read_csv(input_filename)

	def writetocsv(self, conf_filename):
		self.conf.to_csv(conf_filename, index = False)

	def loadrem(self):
		self.rem = pd.read_csv(self.output_filename, sep='\t', header=None)
		# adding a fifth column as the SINR in dB
		self.rem.loc[:,4] = 10*np.log(self.rem.loc[:,3])

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
		ax.scatter(self.rem.loc[:,0], self.rem.loc[:,1], s=25,\
				c = self.rem.loc[:,4], marker='s', edgecolors = 'none')
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

	def plot_ue_dist(self):
		fig = plt.figure(figsize=(7,8))
		ax = fig.add_subplot(111)
		ax.plot(self.ue_dist.x, self.ue_dist.y, 'x')
		ax.set_xlim(self.x_min, self.x_max)
		ax.set_ylim(self.y_min, self.y_max)
		plt.show()


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
