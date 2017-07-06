from environment import network
import random
import pandas as pd

env = network()
print('loading from csv....')
env.readfromcsv()

# creates an empty dataframe with four columns: S, A, R, S'
df = pd.DataFrame()

# gets the initial state
state = env.getstate_vector()

N = 500 # number of steps
for i in range(N):
	action = random.randrange(126)
	new_state, reward = env.execute_action(action)
	print('state_vector: {}'.format(new_state))
	print('reward: {}'.format(reward))
	# appends new data to the dataframe
	df = df.append({'state': state, 'action': action, 'reward': reward, 'new_state': new_state},
			  ignore_index=True)
	# updates the state
	state = new_state

# print(df)
df.to_csv('dataset.csv', index=False)