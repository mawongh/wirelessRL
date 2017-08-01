# the usual imports
# import pandas as pd
import numpy as np
# from scipy import spatial
# importing the network environment class
from environment import network
# to scale the feature vectors
from sklearn.preprocessing import StandardScaler
# additional import for the Neural Network
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers

from linear_aproximation import e_greedy_timedecay


class Model:
	def __init__(self, alpha = 0.001, state_vector_len = 105, action_len = 126, max_experiences = 1000, batch_sz = 128):
		# creates the experiende replay memory D
		self.max_experiences = max_experiences
		self.batch_sz = batch_sz
		self.D = {'s': [], 'a': [], 'r': [], 's_next': [], 'terminal': []}
		# the number of hidden neurons
		N = 200
		# create the graph for the state-action value funtion Q
		self.Q = Senquential()
		# input layer
		self.Q.add(Dense(N, activation = 'relu', input_shape=(state_vector_len,)))
		# second layer hidden
		self.Q.add(Dense(N, activation = 'relu'))
		# output layer
		self.Q.add(Dense(action_len, activation = 'linear'))
		# defines the optimizer
		opt = optimizers.SGD(lr = alpha)
		# compiles the model
		self.Q.compile(optimizer=opt, loss='mean_squared_error')

		# creates the graph for the target state-action value funtion Qhat
		self.Qhat = Senquential()
		# input layer
		self.Qhat.add(Dense(N, activation = 'relu', input_shape=(state_vector_len,)))
		# second layer hidden
		self.Qhat.add(Dense(N, activation = 'relu'))
		# output layer
		self.Qhat.add(Dense(action_len, activation = 'linear'))
		# compiles the model
		# self.Qhat.compile(optimizer='sgd', loss='mean_squared_error')
		# copies the network weights into the target network
		self.Qhat.set_weights(self.Q.get_weights)

		# creates the scaler for normalizing the raw state  vector
		self.scaler = StandardScaler()
		self.scaler.mean_ = np.loadtxt('s_raw_mean_.csv', delimiter=',')
		self.scaler.scale_ = np.loadtxt('s_raw_scale_.csv', delimiter=',')

	def store_transition(self, s, a, reward, s_next, terminal):
		# normalizes the space vectors
		s = self.scaler.transform(s)
		s_next = self.scaler.transform(s_next)

		if len(self.D['s'] > = self.max_experiences):
			self.D['s'].pop(0)
			self.D['a'].pop(0)
			self.D['r'].pop(0)
			self.D['s_next'].pop(0)
			self.D['terminal'].pop(0)
		self.D['s'].append(s)
		self.D['a'].append(a)
		self.D['r'].append(reward)
		self.D['s_next'].append(s_next)
		self.D['terminal'].append(terminal)

	def sample_minibatch(self):
		# in case the current experience D is less than the batch size
		self.minibatch_sz = np.min([len(self.D['s'], self.batch_sz)])
		# get the indices of the sample
		idx = np.random.choice(len(self.D['s']), size = self.minibatch_sz, replace= False)
		self.minibatch_s = np.array([np.array(self.D['s'][i]) for i in idx])
		self.minibatch_a = [int(self.D['a'][i]) for i in idx]
		self.minibatch_r = [self.D['r'][i] for i in idx]
		self.minibatch_s_next = [self.D['s_next'][i] for i in idx]
		self.minibatch_terminal = [self.D['terminal'][i] for i in idx]

	def set_y(self, gamma = 0.2):
		# generates the prediction
		self.minibatch_y = self.Q.predict(self.minibatch_s)
		argmaxQ = np.max(self.Qhat.predict(self.minibatch_s_next, axis=1))
		# will generate the y acording to the dqn algorithm
		for j in np.arange(len(self.minibatch_s)):
			if self.minibatch_terminal[j] == True:
				target = self.minibatch_r[j]
			else:
				target = self.minibatch_r[j] + gamma * argmaxQ[j]
			self.minibatch_y[j,self_minibatch_a] = target

	def gradient_descent_step(self):
		self.Q.fit(self.minibatch_s, self.minibatch_y, batch_size = self.minibatch_sz)



	# def hat(self, s, a):
	# 	x = self.sa2x_v1(s, a)
	# 	return self.theta.dot(x)

	# def grad(self, s, a):
	# 	# since it is a linear aproximation, the gradient is the same feature vector
	# 	return self.sa2x_v1(s, a)
	@staticmethod
	def getQs(nn, s):
		s = s.reshape(1, -1)
		return nn.predict_on_batch(s)[0]


def main():
	# number of episodes M
	M = 2
	# number of steps per episodes T
	T = 10
	# number of steps C to copied weights into target network
	C = 50

	alpha = 0.001
	gamma = 0.5
	r_thershold = 5

	outputfile = 'DQN_training_history.csv'
	# for logging creates an empty dataframe with four columns: S, A, R, S'
	df = pd.DataFrame(columns = ['00_episode', '01_step',
								'state', 'action', 'reward',
								'new_state', 'epsilon', 'terminal'])
	df.to_csv(outputfile, index=False)

	# creates the environment
	env = network()

	# creates an instance of the model and initizializes the weights (theta)
	agent = Model(alpha = alpha)

	# defining the policy
	e = e_greedy_timedecay(126, initial_epsilon = 1, decaying_factor = 0.9999)

	for episode in np.arange(M):
		# initial state for episode
		env.readfromcsv()
		s = env.getstate_vector2()

		for t in np.arange(T):
			# choose action a
			a = e.get_action(agent.Q, s)
			# take action a, observe r, s'
			s_next, reward = env.execute_action(a)
			# check if next state is terminal
			if reward >= r_thershold:
				terminal = True
			else:
				terminal = False
			# store transition in D
			agent.store_transition(s, a, reward, s_next, terminal)
			# sample random mini-batch of transition from D
			agent.sample_minibatch()
			# set y_j if step is terminal or not
			agent.set_y(gamma = gamma)
			# Perform a gradient descent step with respect to the network parameters
			agent.gradient_descent_step()
			# Every C steps copies target network
			if (t % C) == 0:
				agent.Qhat.set_weights(agent.Q.get_weights())
			# decay epsilon
			e.decay()
			# stores the new step information
			# appends step information to the dataframe
			record = df.append({'00_episode': episode, '01_step': t,
							'state': s, 'action': a, 'reward': reward,
							'new_state': s_next, 'epsilon': e.epsilon,
							'terminal': terminal},
							ignore_index=True)
			# s = s'
			s = s_next

			# saves the dataframe for further analysis
			record.to_csv(outputfile, mode='a', header=False, index=False)





		
