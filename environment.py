import pandas as pd
import numpy as np
import subprocess
import matplotlib.pyplot as plt

class network():
	default_remfile = 'lena-simple_env.rem'
	tempfile = 'lena-simple_env_temp.csv'

	def __init__(self, output_filename = default_remfile):
		self.output_filename = output_filename

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
		plt.figure(figsize=(9,8))
		plt.scatter(self.rem.loc[:,0], self.rem.loc[:,1],\
					c = self.rem.loc[:,4], marker='s', edgecolors='none')
		plt.colorbar()
		plt.show()

	def cdfrem(self):
		pass

def test_network_class():
	conf_file = 'simple_env_conf.csv'
	
	print('creating the class...')
	net1 = network()

	print('reading the class from file...')
	net1.readfromcsv(conf_file)

	print('running the simulation...\n')
	net1.runsimulation()

	print('plot the output remfile...')
	net1.plotrem()

if __name__ == '__main__':
	# runs the test_network_class function
	test_network_class()
