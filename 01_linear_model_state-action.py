# the required imports
import numpy as np
import pandas as pd
from linear_aproximation import Model
from environment import network
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

path = '/Users/mawongh/OneDrive/REFERENCE FILE/D/Disertation/brainstorming/'
dataset = pd.read_pickle(path + 'full_dataset.pickle')

np.random.seed(2017)
N = len(dataset)
sample_sizes = np.array([N * .25, N * .50, N * .75]).astype(int)
sample_indexes = [np.random.choice(np.arange(N), size = sz, replace=False)
                 for sz in sample_sizes]
# np.random.choice(np.arange(10), size =5, replace=False)
datasets = [dataset.iloc[idx] for idx in sample_indexes]

# model 3 - full linear model with domain knowledge inclusion

ds_idx =2
# # Instantiate the model that includes the state to features implementation of the function
model = Model()

print('Constructing the features....')
X = np.array([model.sa2x_v1(datasets[ds_idx].state[i], int(datasets[ds_idx].action[i])) 
              for i in datasets[ds_idx].index])
y = datasets[ds_idx].reward.values

# gets the training and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.15, random_state=42)

# the linear model
reg = linear_model.SGDRegressor(alpha = 0.001, n_iter = 50)
print('fitting model 3...')
print(reg)
reg.fit(X_train, y_train)
# print(reg.coef_)

# Do the prediction and calculate the performance (MSE) for model 1
# Xtest_transformed = scaler.transform(X_test)
# x = np.arange(len(Xtest_transformed))
x = np.arange(len(X_test))
y_hat = reg.predict(X_test)
plt.plot(x, y_test)
plt.plot(x, y_hat)
plt.show()

y_train_hat = reg.predict(X_train)
train_MSE = mean_squared_error(y_train, y_train_hat)
print('train set MSE: {}'.format(train_MSE))

test_MSE = mean_squared_error(y_test, y_hat)
print('test set MSE: {}'.format(test_MSE))
