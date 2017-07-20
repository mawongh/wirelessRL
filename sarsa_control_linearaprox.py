# the usual imports
import pandas as pd
import numpy as np

# imports for the model
from linear_aproximation import Model, e_greedy_timedecay


# importing the network environment class
from environment import network

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
	e = e_greedy_timedecay(126, initial_epsilon = 0.9, decaying_factor = 0.999)

	# creates an empty dataframe with four columns: S, A, R, S'
	df = pd.DataFrame(columns = ['00_episode', '01_step',
								'state', 'action', 'reward',
								'new_state', 'new_action',
								'x', 'theta'])
	df.to_csv(outputfile, index=False)


	for episode in range(N_episodes):
		# initial state and action for episode
		env.readfromcsv()
		# env.reset()
		s = env.getstate_vector2()
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
							'x': q.sa2x_v1(s,a), 'theta': q.theta},
							ignore_index=True)
			

			# saves the dataframe for further analysis
			record.to_csv(outputfile, mode='a', header=False, index=False)

			s = new_s
			a = new_a

			step += 1

		# end while (steps)

	# end for (episodes)

# end __main__






