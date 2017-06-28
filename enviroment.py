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

	def plotrem(self):
		rem = pd.read_csv(self.output_filename, sep='\t', header=None)
		plt.figure(figsize=(9,8))
		plt.scatter(rem.loc[:,0], rem.loc[:,1], c = 10*np.log(rem.loc[:,3]), marker='s', edgecolors='none')
		plt.colorbar()
		plt.show()


# test for the use of the class
# infile = 'simple_env_conf.csv'
# outfile = 'temp_conf.csv'
# # outfile = 'lena-simple_env.rem'

# print('creating the class...')
# net1 = network()

# print('reading the class from file...')
# # net1.readfromcsv(infile)

# # print('writing to a sample file...')
# # net1.writetocsv(outfile)

# # print(net1.output_filename)
# # print('running the simulation...\n')
# # net1.runsimulation()

# # plot the output remfile
# net1.plotrem()
