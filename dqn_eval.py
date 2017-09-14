# the usual imports
# import pandas as pd
import numpy as np
import pandas as pd
# from scipy import spatial
# importing the network environment class
from environment import network
# to scale the feature vectors
from sklearn.preprocessing import StandardScaler
# additional import for the Neural Network
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras import optimizers


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
			Qs = model.predict(state_vector.reshape(1,-1))[0]
			# print(Qs)
			action = np.argmax(Qs)
			# print('action: {}'.format(action))
		else: # it will explore
			print('explore..')
			action = np.random.choice(self.action_space)
			# print('action: {}'.format(action))
		return action


def main():
	# number of episodes M
	M = 20
	# M = 1
	# number of steps per episodes T
	T = 500
	# T = 5

	r_thershold = 2

	outputfile = 'dqn_B_eval_history.csv'
	# for logging creates an empty dataframe with four columns: S, A, R, S'
	df = pd.DataFrame(columns = ['00_episode', '01_step',
								'state', 'action', 'reward',
								'new_state', 'epsilon', 'terminal'])
	df.to_csv(outputfile, index=False)

	# creates the environment
	env = network()


	# defines the model
	# model 6
	agent = Sequential()
	n_cols = 105

	agent.add(Dense(400, activation = 'relu', input_shape=(n_cols,)))
	agent.add(Dense(400, activation = 'relu'))
	agent.add(Dense(400, activation = 'relu'))
	agent.add(Dense(126, activation = 'linear'))
	# model.compile(optimizer=optimizers.Adam(), loss='mean_squared_error')
	# loads the model from file
	model_weigths = '300K_Q_network_final_weights_v2.h5'
	agent.load_weights(model_weigths)

	scaler = StandardScaler()
	scaler.mean_ = np.loadtxt('170908_DQN_300K_scaler-mean_.csv', delimiter=',')
	scaler.scale_ = np.loadtxt('170908_DQN_300K_scaler-scale_.csv', delimiter=',')


	# defining the policy
	e = e_greedy_timedecay(126, initial_epsilon = 0.1, decaying_factor = 1)

	for episode in np.arange(M):
		# initial state for episode
		env.readfromcsv()
		s_wo_Scale = env.getstate_vector2()
		s = scaler.transform(s_wo_Scale.reshape(1, -1))

		for t in np.arange(T):
			print('episode: {} , step: {}'.format(episode, t))
			# choose action a
			a = e.get_action(agent, s)
			# take action a, observe r, s'
			s_next_wo_Scale, reward = env.execute_action(a)
			s_next = scaler.transform(s_next_wo_Scale.reshape(1, -1))
			# check if next state is terminal
			if reward >= r_thershold:
				terminal = True
			else:
				terminal = False

			print('reward: {}'.format(reward))
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
		# end for

	# end for

# end main()

# main loop
if __name__ == '__main__':
	main()





		
