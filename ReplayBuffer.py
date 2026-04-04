import numpy as np
import random

#code a replay buffer
class ReplayBuffer:
	def __init__(self, max_size=100000):
		self.buffer = []
		self.max_size = max_size
		self.size = 0

	def add(self, experience):
		self.buffer.append(experience)
		self.size += 1
		if self.size > self.max_size:
			self.buffer.pop(0)
			self.size -= 1
	
	def add_batch(self, experiences):
		self.buffer.extend(experiences)
		self.size += len(experiences)
		while self.size > self.max_size:
			self.buffer.pop(0)
			self.size -= 1
		

	def sample(self, batch_size):
		samples = random.sample(self.buffer, batch_size)
		# return map(list, zip(*samples))
		return samples

	def size(self):
		return self.size

	def clear(self):
		self.buffer = []
		self.size = 0

#Prioritized replay buffer
class PrioritizedReplayBuffer(ReplayBuffer):
	def __init__(self, max_size=100000):
		super(PrioritizedReplayBuffer, self).__init__(max_size=max_size)
		self.priorities = np.zeros((max_size,), dtype=np.float32)
		
	def add(self, experience):
		max_prio = self.priorities.max() if self.buffer else 1.0
		super().add(experience)
		self.priorities[self.size-1] = max_prio

	def sample(self, batch_size, beta=0.4):
		if self.size == self.max_size:
			prios = self.priorities
		else:
			prios = self.priorities[:self.size]
		probs = prios ** beta
		probs /= probs.sum()
		indices = np.random.choice(self.size, batch_size, p=probs)
		samples = [self.buffer[idx] for idx in indices]
		total = self.size
		weights = (total * probs[indices]) ** (-beta)
		weights /= weights.max()
		return samples, indices, weights
	
	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = prio

	def clear(self):
		super().clear()
		self.priorities = np.zeros((self.max_size,), dtype=np.float32)

