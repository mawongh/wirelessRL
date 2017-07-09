# the usual imports
import pandas as pd
import numpy as np

# importing the network environment class
from environment import network

class Model:
	def __init__(self):
		self.theta = np.random.randn(169) / 2 # initializes the weights

	# the feature construction
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

	def hat(self, s, a):
		x = self.sa2x(s, a)
		return self.theta.dot(x)

	def grad(self, s, a):
		# since it is a linear aproximation, the gradient is the same feature vector
		return self.sa2x(s, a)

def getQs(model, s):
	Qs = {}
	for a in range(126):
		q_sa = model.hat(s, a)
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
		print('rd: {}, eps: {}'.format(rd, e.epsilon))
		if rd >= self.epsilon: # if will exploit
			print('exploit..')
			Qs = getQs(model, state_vector)
			# print(Qs)
			action = max(Qs, key=Qs.get)
			# print('action: {}'.format(action))
		else: # it will explore
			print('explore..')
			action = np.random.choice(self.action_space)
			# print('action: {}'.format(action))
		return action


# main loop
if __name__ == '__main__':
	# creates the environment
	env = network()
	env.readfromcsv()

	# Variable initizialization
	N_episodes = 10
	max_steps_per_episode = 200
	alpha = 0.001
	gamma = 0
	r_thershold = 5
	outputfile = 'SARSA_training_history.csv'

	# creates an instance of the model and initizializes the weights (theta)
	q = Model()

	# defining the policy
	e = e_greedy_timedecay(126, initial_epsilon = 1, decaying_factor = 1)

	# creates an empty dataframe with four columns: S, A, R, S'
	df = pd.DataFrame(columns = ['00_episode', '01_step',
								'state', 'action', 'reward',
								'new_state', 'new_action',
								'x', 'theta'])
	df.to_csv(outputfile, index=False)


	for episode in range(N_episodes):
		# initial state and action for episode
		env.reset()
		s = env.getstate_vector()
		a = e.get_action(q, s)

		# initialize the steps
		step = 0

		# repeat (for each step of episode)
		while step < max_steps_per_episode:

			print('episode: {} , step: {}'.format(episode, step))

			# print(q.theta)	
			# decay epsilon
			e.decay()
			# take action a, observe R, S'
			new_s, reward = env.execute_action(a)
			print('action: {}, reward: {}'.format(a, reward))
			# if S' is terminal
			if reward > r_thershold:
				# update theta
				print('Terminal state!!!!!!')
				q.theta += alpha*(reward - q.hat(s, a))*q.grad(s,a)
				# go to next episode
				break
			# choose A'as function of qhat(S,., theta)
			new_a = e.get_action(q, s)
			# update theta
			q.theta += alpha*(reward + gamma*q.hat(new_s, new_a) - q.hat(s, a))*q.grad(s,a)

			# appends step information to the dataframe
			record = df.append({'00_episode': episode, '01_step': step,
							'state': s, 'action': a, 'reward': reward,
							'new_state': new_s, 'new_action': new_a,
							'x': q.sa2x(s,a), 'theta': q.theta},
							ignore_index=True)
			

			# saves the dataframe for further analysis
			record.to_csv(outputfile, mode='a', header=False, index=False)

			s = new_s
			a = new_a

			step += 1

		# end while (steps)

	# end for (episodes)

# end __main__






