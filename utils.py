import numpy as np
import tensorflow as tf
from fairness_functions import fairness_router

class EpsilonDecay():
	"""
	Wrapper class to handle epsilon decay
	"""
	def __init__(self, start, end, decay_rate, greedy=False):
		self.start = start
		self.end = end
		self.decay_rate = decay_rate
		self.current = start
		self.greedy = greedy

		if self.greedy:
			self.current = 0
	
	def get(self):
		return self.current
	
	def reset(self):
		self.current = self.start
		if self.greedy:
			self.current = 0
		return self.current
	
	def decay(self, eps=None):
		if eps is None:
			self.current = max(self.end, self.current*self.decay_rate)
		else:
			self.current = eps
		return self.current


def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)



#################### Logging ####################
def add_epi_metrics_to_logs(summary_writer, rewards, losses, beta, i_episode, max_steps, verbose=False, prefix="", logging=True, fair_rewards=None, fairness_type="variance", fairness_kwargs={}, fairness_function=None):
	#Add episode's metrics to logs
	#rewards is a list of utilities for each agent
	system_utility = np.sum(rewards)
	if fair_rewards is None:
		fair_rewards = rewards
	variance = np.var(fair_rewards)
	min_utility = min(fair_rewards)
	# fairness = -variance/(np.mean(fair_rewards)+0.0001)
	# objective = system_utility - beta*variance
	if fairness_function is None:
		fairness_function = get_fairness_function(fair_rewards, fairness_type, **fairness_kwargs)
	fairness = fairness_function.get_metric(fair_rewards)
	# objective = system_utility + beta*fairness # Not 0-1 beta
	objective = (1-beta)*system_utility + beta*fairness # 0-1 beta

	if verbose:
		print(rewards)
		print("Ep {:>5d} | Objective   {:>5.2f} | Beta {:>5.4f}".format(i_episode, objective, beta))
		print("Ep {:>5d} | Utility     {:>5.2f} | Variance {:>5.2f}".format(i_episode, system_utility, variance))
		print("Ep {:>5d} | Min Utility {:>5.2f} | Fairness({}) {:>5.2f}".format(i_episode, min_utility, fairness_type, fairness))

	if logging:
		with summary_writer.as_default():
			tf.summary.scalar(prefix+"Utility", float(system_utility), step=i_episode)
			tf.summary.scalar(prefix+"Fairness", float(fairness), step=i_episode)
			tf.summary.scalar(prefix+"Min Utility", float(min_utility), step=i_episode)
			tf.summary.scalar(prefix+"Variance", float(variance), step=i_episode)
			tf.summary.scalar(prefix+"Objective", float(objective), step=i_episode)
			
			if losses is not None:
				for key, value in losses.items():
					tf.summary.scalar(prefix+key, float(value), step=i_episode)
	
	metrics = {'system_utility': system_utility, 'fairness': fairness, 'min_utility': min_utility, 'variance': variance, 'objective': objective}
	return metrics
		

def add_metric_to_logs(summary_writer, metric, name, i_episode, verbose=False, logging=True):
	if verbose:
		print(name, metric)
	if logging:
		with summary_writer.as_default():
			tf.summary.scalar(name, float(metric), step=i_episode)

def get_fairness_function(fair_rewards, fairness_type, **kwargs):
	fairness_func = fairness_router(fairness_type, **kwargs)
	return fairness_func
	

def get_metrics_from_rewards(rewards, beta, fair_rewards=None, fairness_type="variance", fairness_kwargs={}, fairness_function=None):
	#rewards is a list of utilities for each agent
	system_utility = np.sum(rewards)
	if fair_rewards is None:
		fair_rewards = rewards
	variance = np.var(fair_rewards)
	min_utility = min(fair_rewards)
	if fairness_function is None:
		fairness_function = get_fairness_function(fair_rewards, fairness_type, **fairness_kwargs)
	fairness = fairness_function.get_metric(fair_rewards)
	objective = (1-beta)*system_utility + beta*fairness # 0-1 beta
	# objective = system_utility + beta*fairness # Not 0-1 beta
	metrics = {
		'system_utility': system_utility,
		'fairness': fairness,
		'min_utility': min_utility,
		'variance': variance,
		'objective': objective
	}
	return metrics

