"""
Torch versions of the agents.
"""
import sys
import copy
import torch
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
from ReplayBuffer import ReplayBuffer, PrioritizedReplayBuffer

class metaNetwork(nn.Module):
	"""
	Boilerplate functions
	"""
	def __init__(self):
		super(metaNetwork, self).__init__()
	
	def set_weights(self, weights):
		self.load_state_dict(weights)
	
	def get_weights(self):
		return self.state_dict()
	
	def save(self, path):
		torch.save(self.state_dict(), path)

	def load(self, path):
		self.load_state_dict(torch.load(path))

class ValueNetwork(metaNetwork):
	"""
	A value network that predicts the value of a state.
	"""
	def __init__(self, num_features, hidden_size):
		super(ValueNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class PPOValueNetwork(metaNetwork):
	"""
	A PPO network that outputs a value.
	Predict a mean and variance, and sample from a normal distribution.
	"""
	def __init__(self, num_features, hidden_size):
		super(PPOValueNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.mean = nn.Linear(hidden_size, 1)
		self.var = nn.Linear(hidden_size, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		mean = self.mean(x)
		var = F.softplus(self.var(x)) # Ensure variance is positive
		return mean, var
	

class A2CNetwork(metaNetwork):
	"""
	A2C implementation. 
	"""
	def __init__(self, num_features, hidden_size, num_actions):
		super(A2CNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.mean = nn.Linear(hidden_size, num_actions)
		self.var = nn.Linear(hidden_size, num_actions)
		self.value = nn.Linear(hidden_size, 1)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		means = self.mean(x)
		vars = F.softplus(self.var(x))
		value = self.value(x)
		return means, vars, value


class PolicyNetwork(metaNetwork):
	"""
	A policy network that outputs a probability distribution over actions.
	"""
	def __init__(self, num_features, hidden_size, num_actions):
		super(PolicyNetwork, self).__init__()
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_actions)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return F.softmax(x, dim=1)

class MultiHeadValueNetwork(metaNetwork):
	"""
	Value network with two heads, one for fairness and one for utility
	"""
	def __init__(self, num_features, hidden_size, learning_beta=0.0):
		super(MultiHeadValueNetwork, self).__init__()
		self.learning_beta = learning_beta
		self.eval_beta = learning_beta
		self.fc1 = nn.Linear(num_features, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fairness_head = nn.Linear(hidden_size, 1)
		self.utility_head = nn.Linear(hidden_size, 1)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		fairness = self.fairness_head(x)
		utility = self.utility_head(x)
		mult = 0 if self.learning_beta == 0 else self.eval_beta/self.learning_beta
		return utility + mult*fairness
	
	def get_util(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		utility = self.utility_head(x)
		return utility
	
	def get_fair(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		fairness = self.fairness_head(x)
		return fairness
	

"""
Agents:
"""

class Agent:
	def __init__(
			self,
			env,
			num_features,
			hidden_size,
			num_actions,
			learning_rate=0.001,
			GAMMA=0.99,
			learning_beta=0.0,
	):
		assert learning_beta >= 0 and learning_beta <= 1, "Learning beta must be between 0 and 1" # 0-1 beta
		self.env = env
		self.num_features = num_features
		self.hidden_size = hidden_size
		self.num_actions = num_actions
		self.learning_rate = learning_rate
		self.GAMMA = GAMMA
		self.learning_beta = learning_beta
		self.eval_beta = learning_beta
		# For tiered behavior
		self.greedy_q = None
		self.use_fair_obs = False

	def get(self, state):
		pass

	def update(self):
		pass

	def insert_greedy_qvals(self, env, obs):
		if self.greedy_q is not None:
			obs_greedy = env.get_obs(use_fair_obs=False)
			#compute q values with the greedy net and extend to each state
			with torch.no_grad():
				greedy_qs = self.greedy_q.get_QValues(self.greedy_q.env, obs_greedy[0])
				for i in range(len(obs[0])):
					obs[0][i] = np.concatenate((obs[0][i], greedy_qs[i]))
		return obs


class DoubleDQNAgent(Agent):
	"""
	This agent uses a value network to predict Q values of post-decision states.
	Uses double DQN to stabilize learning.
	Key idea: during the update, pick a* = argmax_a Q(s', a) using the online network,
	but calculate the target value using the target network Q'(s',a*).
	YDoubleDQN_t = R_(t+1) + Q(S_(t+1), argmax_a Q(St+1 a; w) w_t ) 
	Follows: Deep Reinforcement Learning with Double Q-learning (Hasselt et al. 2015)
	"""
	def __init__(
		self,
		env,
		num_features,
		hidden_size,
		learning_rate=0.001,
		replay_buffer_size=1000000,
		GAMMA=0.99,
		learning_beta=0.0,
	):
		super(DoubleDQNAgent, self).__init__(env, num_features, hidden_size, None, learning_rate, GAMMA, learning_beta)

		self.replay_buffer = ReplayBuffer(replay_buffer_size)
		self.q_network = ValueNetwork(num_features, hidden_size)
		self.target_q_network = ValueNetwork(num_features, hidden_size)
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
		self.loss = nn.MSELoss()

	def get(self, state, target=False):
		state = Variable(torch.from_numpy(state).float().unsqueeze(0))
		if target:
			return self.target_q_network(state)
		return self.q_network(state)
	
	def gets(self, states, target=False):
		states = Variable(torch.from_numpy(states).float())
		if target:
			return self.target_q_network(states)
		return self.q_network(states)
	
	def get_QValues(self, env, states, target=False):
		Qvals = np.zeros((len(states), env.n_actions))
		for i in range(len(states)):
			for j in range(env.n_actions):
				s_i = env.get_post_decision_state_agent(states[i], j, i)
				Qvals[i][j] = float(self.get(np.array([s_i]), target=target))
		return Qvals

	def add_experience(self, prev_state, actions, rewards, f_rewards, new_state, done):
		self.env.set_state(prev_state)
		obs = self.env.get_obs(use_fair_obs=self.use_fair_obs)
		obs = self.insert_greedy_qvals(self.env, obs)

		post_decision_state = self.env.get_post_decision_states(obs, actions)
		# post_decision_state = self.env.get_post_decision_states(prev_state, actions)
		# M.get_post_decision_states(obs, actions)
		experience = {
			'pd_state': copy.deepcopy(post_decision_state),
			'rewards': copy.deepcopy(rewards),
			'f_rewards': copy.deepcopy(f_rewards),
			'new_state': copy.deepcopy(new_state),
			'done': done,
		}
		self.replay_buffer.add(experience)
	
	def update_from_targets(self, states, target_values_util, target_values_fair, actions):
		losses = []
		values = self.gets(states).squeeze()
		target_values = (1-self.learning_beta)*target_values_util + self.learning_beta*target_values_fair # 0-1 beta
		# target_values = target_values_util + self.learning_beta*target_values_fair # Not 0-1 beta
		target_values = Variable(torch.from_numpy(target_values).float())
		loss = self.loss(values, target_values)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		losses.append(loss.item())
		return losses
	
	def _update(self, experiences):
		losses = []
		for experience in experiences:
			pd_state, rewards, f_rewards, new_state, done = experience['pd_state'], experience['rewards'], experience['f_rewards'], experience['new_state'], experience['done']
			n_agents = len(pd_state)
			self.env.set_state(new_state)
			succ_obs = self.env.get_obs(use_fair_obs=self.use_fair_obs)
			succ_obs = self.insert_greedy_qvals(self.env, succ_obs)

			# Compute the optimal actions using the online q-network
			opt_actions = self.env.compute_best_actions(self, self.env, succ_obs)
			new_pd_states = self.env.get_post_decision_states(succ_obs, opt_actions)

			td_rewards = np.array(rewards)
			td_f_rewards = np.array(f_rewards)
			total_rewards = (1-self.learning_beta)*td_rewards + self.learning_beta * td_f_rewards # 0-1 beta
			# total_rewards = td_rewards + self.learning_beta * td_f_rewards # Not 0-1 beta

			states = np.array([pd_state[i] for i in range(n_agents)])
			
			values = self.gets(states).squeeze()
			target_values = total_rewards + (int(not(done)))*self.GAMMA * self.gets(np.array(new_pd_states), target=True).detach().numpy().flatten()
			target_values = Variable(torch.from_numpy(target_values).float())
			
			loss = self.loss(values, target_values)
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			losses.append(loss.item())
		return losses

	def update(self, num_samples, num_min_samples=100000):
		if self.replay_buffer.size < num_min_samples:
			return []
		self.env.reset()
		experiences = self.replay_buffer.sample(num_samples)
		
		losses = self._update(experiences)
		return losses
	
	def update_from_experience(self, experiences):
		self.env.reset()
		return self._update(experiences)

	def update_target_network(self):
		self.target_q_network.set_weights(self.q_network.get_weights())

	def save_model(self, path):
		self.q_network.save(path)

	def load_model(self, path):
		self.q_network.load(path)
		self.update_target_network()

class SplitDoubleDQNAgent(Agent):
	def __init__(
		self,
		env,
		num_features,
		hidden_size,
		learning_rate=0.001,
		replay_buffer_size=1000000,
		GAMMA=0.99,
		learning_beta=0.0,
		learn_fairness=True,
		learn_utility=True,
	):
		super(SplitDoubleDQNAgent, self).__init__(env, num_features, hidden_size, None, learning_rate, GAMMA, learning_beta)
		self.replay_buffer = ReplayBuffer(replay_buffer_size)
		self.utilAgent = DoubleDQNAgent(env, num_features, hidden_size, learning_rate, replay_buffer_size, GAMMA, learning_beta=0)
		self.fairAgent = DoubleDQNAgent(env, num_features, hidden_size, learning_rate, replay_buffer_size, GAMMA, learning_beta=0)
		self.learn_fairness = learn_fairness
		self.learn_utility = learn_utility

	def get(self, state, beta=None, target=False):
		if beta is None:
			beta = self.learning_beta
		return (1-beta)*self.utilAgent.get(state, target) + beta*self.fairAgent.get(state, target) # 0-1 beta
		# return self.utilAgent.get(state, target) + beta*self.fairAgent.get(state, target) # Not 0-1 beta

	def get_QValues(self, env, states, target=False):
		Qvals = np.zeros((len(states), env.n_actions))
		for i in range(len(states)):
			for j in range(env.n_actions):
				s_i = env.get_post_decision_state_agent(states[i], j, i)
				Qvals[i][j] = float(self.get(np.array([s_i]), target=target))
		return Qvals
	
	def add_experience(self, prev_state, actions, rewards, f_rewards, new_state, done):
		self.env.set_state(prev_state)
		obs = self.env.get_obs(use_fair_obs=self.use_fair_obs)
		obs = self.insert_greedy_qvals(self.env, obs)
		post_decision_state = self.env.get_post_decision_states(obs, actions)
		self.replay_buffer.add({
			'pd_state': post_decision_state,
			'rewards': rewards,
			'f_rewards': f_rewards,
			'new_state': new_state,
			'done': done,
		})
	
	def update_from_targets(self, states, target_values_util, target_values_fair, actions):
		losses = {'util': [], 'fair': []}
		to_update = []
		if self.learn_utility:
			to_update.append('util')
		if self.learn_fairness:
			to_update.append('fair')
		
		for key in to_update:
			agent = {'util': self.utilAgent, 'fair': self.fairAgent}[key]
			target_values = {'util': target_values_util, 'fair': target_values_fair}[key]
			values = agent.gets(states).squeeze()
			loss = agent.loss(values, target_values)
			agent.optimizer.zero_grad()
			loss.backward()
			agent.optimizer.step()
			losses[key].append(loss.item())
		return losses
	
	def _update(self, experiences):
		losses = {'util': [], 'fair': []}
		to_update = []
		if self.learn_utility:
			to_update.append('util')
		if self.learn_fairness:
			to_update.append('fair')
		
		for experience in experiences:
			pd_state, rewards, f_rewards, new_state, done = experience['pd_state'], experience['rewards'], experience['f_rewards'], experience['new_state'], experience['done']
			n_agents = len(pd_state)
			self.env.set_state(new_state)
			succ_obs = self.env.get_obs(use_fair_obs=self.use_fair_obs)
			succ_obs = self.insert_greedy_qvals(self.env, succ_obs)

			# Compute the optimal actions using the online q-network
			opt_actions = self.env.compute_best_actions(self, self.env, succ_obs)
			new_pd_states = self.env.get_post_decision_states(succ_obs, opt_actions)

			td_rewards_util = np.array(rewards)
			td_rewards_fair = np.array(f_rewards)

			states = np.array([pd_state[i] for i in range(n_agents)])
			
			agents = {'util': self.utilAgent, 'fair': self.fairAgent}
			all_td_rewards = {'util': td_rewards_util, 'fair': td_rewards_fair}
			for key in to_update:
				agent = agents[key]
				td_rewards = all_td_rewards[key]
				values = agent.gets(states).squeeze()

				target_values = td_rewards + (int(not(done))*self.GAMMA * agent.gets(np.array(new_pd_states), target=True).detach().numpy().flatten())
				target_values = Variable(torch.from_numpy(target_values).float())
				loss = agent.loss(values, target_values)
				agent.optimizer.zero_grad()
				loss.backward()
				agent.optimizer.step()
				
				losses[key].append(loss.item())
			
		return losses
	
	def update(self, num_samples, num_min_samples=100000):
		loss_logs = {'util': [], 'fair': []}
		if self.replay_buffer.size < num_min_samples:
			return loss_logs
		
		self.env.reset()
		experiences = self.replay_buffer.sample(num_samples)
		loss_logs = self._update(experiences)
		# if self.learn_utility:
		# 	loss_logs['util'] = self.utilAgent.update_from_experience(experiences)
		# if self.learn_fairness:
		# 	fair_experiences = []
		# 	for experience in experiences:
		# 		f_exp = copy.deepcopy(experience)
		# 		f_exp['rewards'] = f_exp['f_rewards']
		# 		fair_experiences.append(f_exp)
		# 	loss_logs['fair'] = self.fairAgent.update_from_experience(fair_experiences)

		return loss_logs

	def update_target_network(self):
		self.utilAgent.update_target_network()
		self.fairAgent.update_target_network()

	def save_model(self, path):
		self.utilAgent.save_model(path + "_util")
		self.fairAgent.save_model(path + "_fair")

	def load_util_model(self, path):
		self.utilAgent.load_model(path)
		self.utilAgent.update_target_network()
	
	def load_fair_model(self, path):
		self.fairAgent.load_model(path)
		self.fairAgent.update_target_network()