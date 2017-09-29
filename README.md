# Reinforcement Learning paradigms in the domain of Self-Organizing Networks
Manuel Wong

### Project Summary
Mobile Network Operators are facing the challenge of improving their operation efficiency and costs during these times of exponential mobile communications growth, Self-Organising Networks (SONs) are an initiative to automate many of the network functionalities to improve its effectiveness. This research explores reinforcement learning algorithms to solve a self-planning use case within SONs, a wireless network environment was set-up using the ns-3/lena simulator to test the algorithms. A model-based with a approximate dynamic programming algorithm outperformed the rest of the tested algorithms (SARSA and DQN). Additionally, an implementation for SON within a live network using this algorithm (ADP) was proposed as part of a digital transformation and SDN/NFV strategy.

### Main files description

- **lena-simple_env.cc**, this is the base ns-3/lena code in C++, it is based on the example provided by Jaume Nin <jnin@cttc.es>, reads the temporary configuration network file **simple_env_conf_temp.csv** created by **environment.py**, executes the network simulation and creates the radio environment map (REM) file (*lena-simple_env.rem*)used to calculate the reward.
- **environment.py**, contain the clases that implement all the wireless network environment implementation
- **simple_env_conf.csv**, the initial wireless network configuration that describe the initial state
- **linear_aproximation.py**, includes the linear model and the feature construction coding used for the linear model approximation of the environment and the value function.
- **sarsa_control_linearaprox.py**, implements the SARSA training algorithm with the live environment.
- **dqn.py**, DQN training with live environment
- **model_eval.py**, does the evaluation of the specific model with the live network.
- **ipython_notebooks**, directory where the ipython notebooks are located, mainly the model-based algorithms that were trained using the off-line dataset from the environment

