import copy
import numpy as np
import matplotlib.pyplot as plt
from matching import get_assignment
from scipy.spatial import distance_matrix
from collections import defaultdict
np.random.seed(0)

from DFRL.common.utils import get_more_obs_com

def get_distance(a,b):
	#Get distance between two points
	return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def get_manhattan_distance(a,b):
	#Get manhattan distance between two points
	return abs(a[0]-b[0])+abs(a[1]-b[1])

class DECAEnvt:
	'''
	Meta class for multi-agent central decision-maker environments

	# self.su should be used to keep track of rewards. (system utility)
	# self.discounted_su should be used for fairness
	'''
	def __init__(self, warm_start=0, past_discount=0.995):
		self.warm_start = warm_start
		self.past_discount = past_discount
		self.state_variables = [] # List of state variable names (str). Will be used to get and set state
		self.observation_space = [] # List of observation variable names (str). Will be used to get observation
		self.fairness_vars = '' # name of the variable to use for fairness rewards
		self.external_trigger = None # For debugging
		self.observation_template = None
		self.scaling_factors = None
	
	def get_observation_template(self):
		if self.observation_template:
			return self.observation_template
		state = self.get_stateful_observation()[0][0]
		template = {}
		ind = 0
		for f in self.observation_space:
			try:
				if len(state[f]):
					template[f] = [ind, ind+len(state[f])]
					ind+=len(state[f])
			except:
				template[f] = [ind, ind+1]
				ind+=1
			
		self.observation_template = template
		return template
	
	def get_fairness_rewards(self):
		#returns the fairness rewards if defined. Use system utility by default
		return getattr(self, self.fairness_vars, self.su)

	def reset(self):
		pass

	def step(self, actions):
		# actions[i] is the action taken by agent i
		pass

	def get_state(self):
		return [getattr(self, var) for var in self.state_variables]
	
	def set_state(self, state):
		for i,var in enumerate(self.state_variables):
			setattr(self, var, state[i])
	
	def get_stateful_observation(self):
		# Return the observation space of each agent
		# Should return a list of state lists + a list of resource lists
		# [[state_i], [otherinfo]]
		pass
	
	def get_fair_obs(self):
		# Get fairness observations for all agents
		# This is currenty based on actual utility, su.
		# Discounted su may be misleading
		
		# Advantages signal
		fair_obs = self.discounted_su
		fair_std = np.std(fair_obs)
		if fair_std==0:
			fair_obs = [0 for _ in fair_obs]
		else:
			fair_obs = (fair_obs - np.mean(fair_obs)) / fair_std
			fair_obs = fair_obs.tolist()
		
		#OVERRIDE: return the actual su
		fair_obs = copy.deepcopy(self.discounted_su)
		# sort
		# fair_obs = sorted(fair_obs)
		# more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
		fair_obs = (fair_obs - np.mean(fair_obs)) / (np.std(fair_obs) + 0.0000000001)

		neighbors = self.neighbors()
		
		all_fair_obs = []
		for i in range(self.n_agents):
			fo_i = []
			fo_i.extend(fair_obs)
			more_obs = get_more_obs_com(True, neighbors, self.discounted_su, i, self.neighbors_size+1)
			fo_i.extend(more_obs)
			all_fair_obs.append(fo_i)
		return all_fair_obs

	def get_obs(self, use_fair_obs=False):
		# a general wrapper which parses stateful observation
		obs = copy.deepcopy(self.get_stateful_observation())
		
		if use_fair_obs:
			fair_obs = self.get_fair_obs()
		agents = []
		for i in range(len(obs[0])):
			h = obs[0][i]
			feats = []
			for f in self.observation_space:
				# if it is an iterable, extend, otherwise append
				try:
					if len(h[f]):
						feats.extend(h[f])
				except:
					feats.append(h[f])
			
			# Add fairness signal
			if use_fair_obs:
				feats.extend(fair_obs[i])
			
			agents.append(feats)
		obs[0] = agents
		return obs

	def render(self):
		pass
	
	def get_post_decision_state_agent(self, obs, action, idx):
		# compte the effect of action on the state of agent idx
		# ensure that the reward is not captured, but the other state elements that change are captured
		# look at self.observation space for reference
		pass

	def get_post_decision_states(self, obs, actions):
		return [self.get_post_decision_state_agent(obs[0][i], actions[i], i) for i in range(self.n_agents)]

	def get_all_pd_states(self, obs):
		all_pd_states = []
		for i in range(self.n_agents):
			agent_pd_states = []
			for j in range(self.n_actions):
				h_post = self.get_post_decision_state_agent(obs[0][i], j, i)
				agent_pd_states.append(h_post)
			all_pd_states.append(agent_pd_states)
		return all_pd_states
		
	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False):
		# targets, n_agents, n_resources, su, should be extracted from envt
		# put a general template here
		pass

	def get_resource_counts(self, envt):
		# return the number of resources each agent can get
		pass

	def get_agent_constraints(self, envt):
		# return the constraints on the agents
		return None

	def compute_neighbors(self):
		# compute the neighbors of each agent
		pass

	def neighbors(self):
		# assert self.compute_neighbors
		# neighbors_dist_order: ranking of agents by distance
		# neighbors_nearby: location of nearby agents in the neighborhood. E.g. in 3x3 grid for Job
		self.compute_neighbors()
		return self.neighbors_dist_order, self.neighbors_nearby

	def compute_allocation(self, Qvals, envt):
		resource_counts = self.get_resource_counts(envt)
		agent_constraints = self.get_agent_constraints(envt)
		actions = get_assignment(Qvals, resource_counts, agent_constraints)
		return actions
	
	def get_illegal_actions(self, idx, actions):
		# return the illegal actions for given agent
		return []

class BiasedDMEnvt(DECAEnvt):
	def __init__(self, n_agents=5, warm_start=0, past_discount=0.995, **kwargs):
		"""
		Agents have different utility they get from resources
		Fairness objective is not for the agents to get the same utility, but for the agents to get the same amount of resources
		No One agent is guaranteed to get the resource in each step
		"""
		super().__init__(warm_start, past_discount)
		self.n_agents = n_agents
		self.agent_scores = [1, 0.8, 0.6, 0.4, 0.2] # The reward each agent gets for picking up the resource

		self.state_variables = ['su', 'discounted_su', 'time', 'resource_rate', 'resources', 'discounted_resources', 'discounted_time' ]
		self.observation_space = ['util','agent_score', 'disc_resource_rate', 'relative_util','about_to_get_resource']
		self.observation_space = ['agent_score', 'disc_resource_rate', 'relative_util','about_to_get_resource']
		self.fairness_vars = 'resource_rate'
		self.n_actions = 2
		self.n_neighbors = 4
		self.neighbors_size = 4 # number of inputs for neighbor fetures
		
		self.reset()

	def reset(self):
		self.su = np.zeros(self.n_agents)
		self.time = 0
		self.resources = np.zeros(self.n_agents)

		w = self.warm_start/10 #width of the warm start randomization
		w = self.warm_start/4 #width of the warm start randomization
		warm_start_resources = np.array([
			self.warm_start + np.random.rand()*w - w/2 for _ in range(self.n_agents)
			])
		# self.resource_rate = np.zeros(self.n_agents) # How many resources did this agent get historically
		self.discounted_resources = warm_start_resources
		self.discounted_time = sum(warm_start_resources)
		if self.warm_start==0:
			warm_start_resources = np.array([0.0 for _ in range(self.n_agents)])
			self.discounted_time = 1
		self.resource_rate = warm_start_resources/self.discounted_time # How many resources did this agent get historically
		self.discounted_su = copy.deepcopy(self.resource_rate)
		
		
	def step(self, actions):
		if sum(actions)>1:
			print("invalid action encountered")
			print(actions)
			exit()
		re = [0]*self.n_agents
		for i in range(self.n_agents):
			re[i] = actions[i]*self.agent_scores[i]

		self.time+=1
		self.discounted_time = self.discounted_time*self.past_discount + 1
		self.su+=np.array(re)
		for i in range(self.n_agents):
			self.resources[i]+=actions[i]
			self.resource_rate[i] = (self.resources[i])/(self.time)
			self.discounted_resources[i] = self.discounted_resources[i]*self.past_discount + actions[i]
		self.discounted_su = self.discounted_resources/self.discounted_time
		return re

	def get_stateful_observation(self):
		# Needs to capture what changes before and after the decision both for util and fairness, as well as what causes it
		agents = []
		mean_fair_util = np.mean(self.discounted_su)
		if mean_fair_util==0:
			relative_utils = [0 for su in self.discounted_su]
		else:
			relative_utils = [su/mean_fair_util - 1 for su in self.discounted_su]
		for i in range(self.n_agents):
			h = {
				"util":self.su[i],
				"resource": self.resources[i],
				"relative_util":relative_utils[i],
				"resource_rate":self.resource_rate[i],
				"disc_resource_rate":self.discounted_su[i],
				"agent_score":self.agent_scores[i],
				"about_to_get_resource":-1,
			}

			#Get info about other agents
			others = []
			# append utils and relative_utils of all other agents
			for j in range(self.n_agents):
				if j!=i:
					others.append([
						# self.su[j],
						relative_utils[j],
						self.resource_rate[j],
						self.agent_scores[j]
					])
			#flatten
			others = [feature for other in others for feature in other]
			h['other_agents'] = others
			agents.append(h)

		return [agents, []]
	
	def render(self):
		pstr = ""
		pstr2 = ""
		for i in range(self.n_agents):
			pstr+="agent "+str(i+1)+"\t"
			pstr2+=str(self.su[i])+"\t"
		print(pstr)
		print(pstr2)

	def get_post_decision_state_agent(self, obs, action, idx):
		s_i = copy.deepcopy(obs)
		obs_template = self.get_observation_template()
		# about to get resource idx
		atgr_idx = obs_template['about_to_get_resource'][0]
		s_i[atgr_idx] = action
		return s_i
	
	def get_resource_counts(self, envt):
		return [self.n_agents-1, 1]

	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False, val=False):
		# greedy strategy: round robin (for fairness)
		n_agents = envt.n_agents
		
		# Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand() for act in range(self.n_actions)] for _ in range(n_agents)]
			return self.compute_allocation(Qvals, envt)
		
		if use_greedy:
			# Fair policy: round robin
			res = [r for r in envt.resources]
			min_idx = res.index(min(res))
			Qvals = [[0, 1] if i==min_idx else [1, 0] for i in range(n_agents)]
		else:
			Qvals = model.get_QValues(self, obs[0])

		return self.compute_allocation(Qvals, envt)
	
	def compute_neighbors(self):
		pseudo_distances = np.zeros((self.n_agents, self.n_agents))
		for i in range(self.n_agents):
			pseudo_distances[i, i] = float('+inf')
		self.neighbors_dist_order = pseudo_distances.argsort()[:,:self.n_neighbors]
		self.neighbors_nearby = [[i for i in range(self.n_neighbors)] for j in range(self.n_agents)]
	
	def get_illegal_actions(self, idx, actions):
		# create a dict map of actions 
		#use defaultdict to avoid key errors
		act_counts = defaultdict(int)
		for act in actions:
			if act in act_counts:
				act_counts[act]+=1
			else:
				act_counts[act]=1
		if act_counts[1]>0:
			return [1]
		return []

class BiasedDMDirectEnvt(BiasedDMEnvt):
	"""
	Exactly the same as BiasedDM, but the fairness is over su
	Reuse most functions
	"""
	def __init__(self, n_agents=5, warm_start=0, past_discount=0.995, **kwargs):
		super().__init__(n_agents, warm_start, past_discount, **kwargs)
		self.fairness_vars = ''
		self.observation_space = ['agent_score', 'relative_util', 'about_to_get_resource']
		self.fairness_vars = 'resources'

	def reset(self):
		super().reset()
		w = self.warm_start/4 #width of the warm start randomization
		self.discounted_su = np.array([self.warm_start + np.random.rand()*w - w/2 for _ in range(self.n_agents)])
		if self.fairness_vars=='resources':
			self.discounted_su = self.discounted_resources
	
	def step(self, actions):
		re = super().step(actions)
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)
		if self.fairness_vars=='resources':
			self.discounted_su = self.discounted_resources
		return re

	
class MatthewEnvt(DECAEnvt):
	def __init__(self, 
		  n_agents, 
		  n_resources, 
		  max_size, 
		  min_size=0.01, 
		  size_update=0.005, 
		  base_speed=0.01, 
		  reallocate=False, 
		  simple_obs=False, 
		  warm_start=0,
		  past_discount=0.995,
		  GAMMA=None
		  ):
		super().__init__(warm_start, past_discount)
		self.n_agents = n_agents
		self.n_resources = n_resources
		self.n_actions = n_resources+1
		self.n_neighbors = 3
		self.neighbors_size = self.n_neighbors
		self.max_size = max_size
		self.min_size = min_size
		self.size_update = size_update
		self.base_speed = base_speed
		self.GAMMA = GAMMA

		self.reset()
		
		self.reallocate = reallocate
		self.simple_obs = simple_obs

		self.state_variables = ['ant', 'resource', 'targets', 'size', 'speed', 'su', 'agent_types', 'discounted_su']
		self.fairness_vars = ''
		self.observation_space = ['loc','size','speed', 'eaten','n_other_free_agents','relative_size', 'relative_su', 'other_agents', 'target_resource']
		self.observation_space = ['loc','size','speed', 'eaten','n_other_free_agents','relative_size', 'relative_su', 'other_agents', 'resources', 'target_resource']
		# !!!TODO: DONT USE EATEN AS OBSERVATION SPACE
		self.observation_space = ['loc','size','speed', 'n_other_free_agents', 'any_available','resources', 'relative_size', 'relative_su', 'other_agents', 'target_resource']#barebones
		if self.simple_obs:
			self.observation_space = ['loc','size','speed','n_other_free_agents','relative_size', 'target_resource']
	
	def reset(self):
		agent_types = [0,0,0,0,0,0,2,2,2,2]
		self.agent_types = agent_types
		ant = []
		size = []
		speed = []
		su = [0]*self.n_agents
		targets = []
		for i in range(self.n_agents):
			ant.append(np.random.rand(2))
			size.append(self.min_size + self.agent_types[i]/50)
			speed.append(self.base_speed + size[i])
			targets.append(None)
		su = np.array(su)

		resource=[]
		for i in range(self.n_resources):
			resource.append(np.random.rand(2))

		
		self.resource = resource
		self.ant = ant
		self.targets = targets
		self.size = size
		self.speed = speed
		self.set_speed_from_sizes()
		self.su = su
		
		w = 5 #width of the warm start randomization
		w = self.warm_start/4
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2 
			for _ in range(self.n_agents)])
		
	
	def set_speed_from_sizes(self):
		for i in range(self.n_agents):
			self.speed[i] = self.base_speed + self.size[i]

	def step(self, actions):
		res_ids = [i-1 for i in actions]
		clear_targets = False
		# Actions just decide mapping of agents to resources
		re = [0]*self.n_agents  #rewards. If an agent picks up a resource, get reward of 1
		re_expected = [0]*self.n_agents  #rewards. If an agent is assigned a resource, get reward of 1*discount^T
		for i in range(self.n_agents):
			if res_ids[i]!=-1:
				self.targets[i] = [res_ids[i]]
				#Add the time to reach the resource to the targets vector
				time_to_reach = get_distance(self.ant[i],self.resource[res_ids[i]])/self.speed[i]
				self.targets[i].append(time_to_reach)
				assert self.GAMMA is not None, "GAMMA must be set for re_expected"
				re_expected[i]=1*self.GAMMA**time_to_reach 

			#Move each agent towards its target resource
			if self.targets[i] is not None:
				#Other agents can't pick up the resources if they are claimed
				#if target is overlapped by the agent, remove it
				if self.targets[i][1]<=1:
					self.ant[i][0] = self.resource[self.targets[i][0]][0]
					self.ant[i][1] = self.resource[self.targets[i][0]][1]
					re[i]=1 #Get reward

					#Reset target resource
					self.resource[self.targets[i][0]]=np.random.rand(2)
					self.size[i]=min(self.size[i]+self.size_update, self.max_size)
					# self.speed[i]=self.base_speed+self.size[i]
					self.targets[i]=None
					clear_targets = True
				else:
					#Move agent towards target resource. Each step, move 1/time_remaining of the way
					self.ant[i][0]+=(self.resource[self.targets[i][0]][0]-self.ant[i][0])/self.targets[i][1]
					self.ant[i][1]+=(self.resource[self.targets[i][0]][1]-self.ant[i][1])/self.targets[i][1]
					self.targets[i][1]-=1
					if get_distance(self.ant[i],self.resource[self.targets[i][0]])<self.size[i]:
						re[i]=1
						#Reset target resource
						self.resource[self.targets[i][0]]=np.random.rand(2)
						self.size[i]=min(self.size[i]+self.size_update, self.max_size)
						# self.speed[i]=self.base_speed+self.size[i]
						self.targets[i]=None
						clear_targets = True
			else:
				#Move randomly
				p_move = 0.8
				dr = np.random.rand()*2*np.pi
				if np.random.rand()<p_move:
					self.ant[i][0]+=np.cos(dr)*self.speed[i]
					self.ant[i][1]+=np.sin(dr)*self.speed[i]
			
			#Check for bounds
			if self.ant[i][0]<0:
				self.ant[i][0]=0
			if self.ant[i][0]>1:
				self.ant[i][0]=1
			if self.ant[i][1]<0:
				self.ant[i][1]=0
			if self.ant[i][1]>1:
				self.ant[i][1]=1
		
		#Update speeds
		self.set_speed_from_sizes()

		# If any resources were picked up, reset the targets
		if self.reallocate:
			if clear_targets:
				# print("Clearing Targets")
				for i in range(self.n_agents):
					self.targets[i]=None

		self.su+=np.array(re) # This always keeps track of what the agent has received exactlys
		
		#Update the discounted su
		re_return = np.array(re_expected)
		# re_return = np.array(re)
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re_return)

		return re_return

	def get_stateful_observation(self):
		distances, nearby = self.neighbors()
		agents = []
		for i in range(self.n_agents):
			h={}
			h['loc'] = [self.ant[i][0], self.ant[i][1]]
			h['size'] = self.size[i]
			h['speed'] = self.speed[i]
			h['eaten'] = self.su[i]
			
			#Get number of agents without a target resource
			n = sum([1 for j in range(self.n_agents) if self.targets[j] is None])
			h['n_other_free_agents'] = n
			h['relative_size'] = self.size[i]/np.mean(self.size) - 1
			mean_disc_su = np.mean(self.discounted_su)
			if mean_disc_su==0:
				h['relative_su'] = 0
			else:
				h['relative_su'] = self.discounted_su[i]/mean_disc_su - 1

			#Get info about other agents
			others = []
			neighbors_only = False
			# # only for the nearest neighbors
			if neighbors_only:
				for j in range(self.n_neighbors):
					others.append(self.ant[distances[i][j]][0])
					others.append(self.ant[distances[i][j]][1])
					others.append(self.size[distances[i][j]])
					others.append(self.speed[distances[i][j]])
					others.append(1 if self.targets[distances[i][j]] is None else 0)
			else:
				# append locations of all other agents, their sizes, and their distances to the target resource
				for j in range(self.n_agents):
					if j!=i:
						others.append([
							self.ant[j][0],self.ant[j][1],
							self.speed[j], 
							self.su[j],
							# get_distance(self.ant[j], self.resource[self.targets[i][0]]) if self.targets[i] is not None else -1,
							1 if self.targets[j] is None else 0
							])
				#flatten
				others = [feature for other in others for feature in other]
			h['other_agents'] = others

			# resources: x,y,dist
			h['resources'] = []
			occupied_resources = [self.targets[j][0] for j in range(self.n_agents) if self.targets[j] is not None]
			for j in range(self.n_resources):
				res = []
				available = -1 if j in occupied_resources else 1
				res.append(available)
				# if not occupied:
				# 	res.append(get_distance(self.ant[i], self.resource[j]))
				# else:
				# 	res.append(100)
				res.append(self.resource[j][0])
				res.append(self.resource[j][1])
				# res.append(get_distance(self.ant[i], self.resource[j]))
				h['resources'].extend(res)
			
			h['can_take_resource'] = 1 if self.targets[i] is None else 0
			h['num_free_resources'] = len([1 for j in range(self.n_resources) if j not in occupied_resources])
			h['any_available'] = 1 if h['num_free_resources']>0 else 0
			h['action'] = [0 for _ in range(self.n_actions)]

			#Get info about target resource
			if self.targets[i] is not None:
				t = [self.resource[self.targets[i][0]][0], self.resource[self.targets[i][0]][1], self.targets[i][1]]
			else:
				t = [-1,-1,100]
			h['target_resource'] = t

			agents.append(h)

		return [agents, copy.deepcopy(self.resource)]

	def get_post_decision_state_agent(self, obs, action, idx):
		s_i = copy.deepcopy(obs)
		obs_template = self.get_observation_template()
		if 'action' in obs_template:
			# s_i[obs_template['action'][0]] = action
			ind = obs_template['action'][0]
			for j in range(self.n_actions):
				s_i[ind+j] = 1 if j==action else 0
		if 'target_resource' in obs_template:
			if 'loc' in obs_template:
				ant_loc = [obs[obs_template['loc'][0]], obs[obs_template['loc'][1]]]
				res_id = action -1 
				if res_id!=-1:
					target_res = self.resource[res_id]
					target = [target_res[0], target_res[1], get_distance(ant_loc, target_res)/self.speed[idx]]
				else:
					target = [-1,-1,100]
				start_idx = obs_template['target_resource'][0]
				for j in range(len(target)):
					s_i[start_idx+j] = target[j]
		return s_i

	def render(self):
		for i in range(self.n_agents):
			theta = np.arange(0, 2*np.pi, 0.01)
			x = self.ant[i][0] + self.size[i] * np.cos(theta)
			y = self.ant[i][1] + self.size[i] * np.sin(theta)
			plt.plot(x, y)
			if self.targets[i] is not None:
				#plot a line from ant to target
				plt.plot([self.ant[i][0],self.resource[self.targets[i][0]][0]],[self.ant[i][1],self.resource[self.targets[i][0]][1]], color = 'red')
		for i in range(self.n_resources):
			plt.scatter(self.resource[i][0], self.resource[i][1], color = 'green')
		plt.axis("off")
		plt.axis("equal")
		plt.xlim(0 , 1)
		plt.ylim(0 , 1)
		plt.ion()
		plt.pause(0.1)
		plt.close()

	def get_resource_counts(self, envt):
		targets, n_agents, n_resources = envt.targets, envt.n_agents, envt.n_resources
		occupied_resources = set([targets[j][0] for j in range(n_agents) if targets[j] is not None])
		
		resource_counts = [n_agents] # First action does not have a resource restriction
		for i in range(n_resources):
			# Add available resources
			if i not in occupied_resources:
				resource_counts.append(1)
			else:
				resource_counts.append(0)
		return resource_counts

	def get_agent_constraints(self, envt):
		# agents with targets can only take the action to do nothing
		agent_constraints = []
		for i in range(envt.n_agents):
			if envt.targets[i] is not None:
				illegal_resources = [j+1 for j in range(envt.n_resources)]
				agent_constraints.append(illegal_resources)
			else:
				agent_constraints.append([])
		return agent_constraints

	def compute_best_actions(self, model, envt, obs, epsilon=0.0, beta=0.0, direction='both', use_greedy=False):
		targets, n_agents, n_resources, su = envt.targets, envt.n_agents, envt.n_resources, envt.su
		occupied_resources = set([targets[j][0] for j in range(n_agents) if targets[j] is not None])
		# random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand()*max(2-ind,1) for ind in range(self.n_actions)] for _ in range(n_agents)] #Increase importance of doing nothing
			return self.compute_allocation(Qvals, envt)
		
		# greedy strategy
		if use_greedy:
			Qvals = [[-1000000 for _ in range(self.n_actions)] for _ in range(n_agents)]
			for i in range(n_agents):
				Qvals[i][0] = 0
				for j in range(n_resources):
					if j not in occupied_resources:
						Qvals[i][j+1] = envt.speed[i]
		else:
			Qvals = model.get_QValues(self, obs[0])

		return self.compute_allocation(Qvals, envt)
	
	def compute_neighbors(self):
		distances = distance_matrix(self.ant, self.ant, p=2)
		for i in range(self.n_agents):
			distances[i,i] = float('+inf')
		distances = distances.argsort()[:,:self.n_neighbors]
		self.neighbors_dist_order = distances
		self.neighbors_nearby = [[i for i in range(self.n_neighbors)] for _ in range(self.n_agents)]

	def get_illegal_actions(self, idx, actions):
		illegal_actions = []
		if self.targets[idx]!=None:
			illegal_actions = [i+1 for i in range(self.n_resources)]
		occupied_resources = set([self.targets[j][0] for j in range(self.n_agents) if self.targets[j] is not None])
		# consider just allocated resources
		for act in actions:
			if act!=None and act!=0:
				occupied_resources.add(act-1)
		for i in range(self.n_actions):
			if i-1 in occupied_resources:
				illegal_actions.append(i)
		return illegal_actions


class JobSchedulingEnvt(DECAEnvt):
	'''
	Many workers that desire to work on a job. As long as worker occupies the job's location, they get a reward
	Central agent approach. The constraint is no worker can be in the same location at the same time. 
	The allowed actions are only the grid locations immediately adjacent to the agent or to stay.

	Actions are converted into integer based on new grid location. 
	Actions that would lead outside the grid: consider separately or ignore?
		Going with ignore, they would map back to the current location.
	'''
	def __init__(self, 
		  n_agents, 
		  gridsize=5,
		  reallocate=True, 
		  simple_obs=False, 
		  warm_start=0,
		  past_discount=0.995,
		  ):
		
		super().__init__(warm_start, past_discount)
		
		self.n_agents = n_agents
		self.gridsize = gridsize
		self.n_actions = 5
		self.n_neighbors = min(3, n_agents-1)
		self.neighbors_size = 8
		
		self.reset()

		self.reallocate = reallocate
		self.simple_obs = simple_obs
		self.state_variables = ['grid', 'ant', 'job', 'su', 'discounted_su']
		# !!WARNING!!: If the observation space is changed, stateful observation must be implemented
		if self.simple_obs:
			# self.observation_space = ['relative_job', 'grid', 'relative_util']
			self.observation_space = ['relative_job', 'grid', 'discounted_su']
	
	def reset(self):
		# The grid is a 2D array of size gridsize x gridsize
		# Each agent is assigned a random location on the grid
		# Empty grid locations are 0
		# Agent locations are 1,2,3,4,5...
		# Job location is always fixed, so not tracked on the grid

		self.grid = np.zeros((self.gridsize, self.gridsize))
		ant = []
		# #Random start locations
		# for i in range(self.n_agents):
		# 	loc = np.random.randint(0,self.gridsize,2)
		# 	while self.grid[loc[0],loc[1]]!=0:
		# 		loc = np.random.randint(0,self.gridsize,2)
		# 	ant.append(loc)
		# 	self.grid[ant[i][0],ant[i][1]]=i+1
		# ant = np.array(ant)

		# self.job = np.random.randint(1,self.gridsize-1,2)
		#place job at the center
		self.job = [self.gridsize//2, self.gridsize//2]
		# place agents at the corners
		ant = []
		for i in range(self.n_agents):
			# pick the ith corner
			loc = [0,0]
			if i==1:
				loc = [0, self.gridsize-1]
			elif i==2:
				loc = [self.gridsize-1, 0]
			elif i==3:
				loc = [self.gridsize-1, self.gridsize-1]
			ant.append(loc)
			self.grid[loc[0],loc[1]]=i+1
		self.job_reward = 1

		self.ant = ant
		self.targets = [None]*self.n_agents
		self.su = np.zeros(self.n_agents)

		w = 5 #width of the warm start distribution
		w = self.warm_start/4
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2
			for _ in range(self.n_agents)])
		
	def map_grid_to_idx(self, i,j):
		return i*self.gridsize+j

	def map_idx_to_grid(self, idx):
		return idx//self.gridsize, idx%self.gridsize

	def pre_move(self, ant, dir):
		x, y = ant
		dir_map = {
			0: [0,0],
			1: [0,1],
			2: [0,-1],
			3: [-1,0],
			4: [1,0]
		}
		delta_x = dir_map[dir][0]
		delta_y = dir_map[dir][1]
		new_x = x + delta_x
		new_y = y + delta_y
		return new_x, new_y
		
	
	def move(self, ant, dir):
		#Move agent i in direction
		#dir is one of [stay, up, down, left, right]
		illegal=False
		x,y = ant
		new_x, new_y = self.pre_move(ant,dir)

		# Simple bounds checking.
		if 0<=new_x<self.gridsize and 0<=new_y<self.gridsize:
			# No need to check if someone else is there. That is handled by the ILP.
			x, y = new_x, new_y
		else:
			illegal=True
		return x,y, illegal
		
	# Each action is one of [stay, up, down, left, right], and is mapped to a unique grid lcoation
	def step(self, actions):
		# actions[i] is the action taken by agent i
		reset_job = False
		re = [0]*self.n_agents  #rewards. If an agent is on the job, get reward of 1
		for i in range(self.n_agents):
			if actions[i]!=-1:
				new_x, new_y, _ = self.move(self.ant[i], actions[i])			
				if self.grid[self.ant[i][0],self.ant[i][1]]==i+1:
					# If the agent is still at the old location, update the grid
					# If another update has already been made, don't update the grid
					self.grid[self.ant[i][0],self.ant[i][1]]=0
				self.ant[i][0] = new_x
				self.ant[i][1] = new_y
				self.grid[new_x, new_y]=i+1
	

			#Check if agent is on the job
			if self.ant[i][0]==self.job[0] and self.ant[i][1]==self.job[1]:
				re[i]=self.job_reward
				# reset_job = True
				
		# if reset_job:
		# 	self.job = np.random.randint(0,self.gridsize,2)	
		self.su+=np.array(re)
		#Update the discounted utilities
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)

		# # Add supplemental reward based on distance from job
		for i in range(self.n_agents):
			re[i] += -0.1*get_distance(self.ant[i], self.job)
		
		return re

	def get_stateful_observation(self):
		agents = []
		for i in range(self.n_agents):
			h={}
			h['loc'] = [self.ant[i][0], self.ant[i][1]]
			h['util'] = self.su[i]
			mean_disc_su = np.mean(self.discounted_su)
			if mean_disc_su==0:
				h['relative_util'] = 0
			else:
				h['relative_util'] = self.discounted_su[i]/mean_disc_su - 1
			h['discounted_su'] = self.discounted_su[i]
			h['su'] = self.su[i]

			#Get info about other agents
			others = []
			# append locations of all other agents
			for j in range(self.n_agents):
				if j!=i:
					others.append([
						self.ant[j][0],self.ant[j][1],
						self.su[j],
					])
			#flatten
			others = [feature for other in others for feature in other]
			h['other_agents'] = others

			# alt: get the 3x3 grid around the agent
			h['grid'] = []
			for x in range(self.ant[i][0]-1, self.ant[i][0]+2):
				for y in range(self.ant[i][1]-1, self.ant[i][1]+2):
					if 0<=x<self.gridsize and 0<=y<self.gridsize:
						h['grid'].append(self.grid[x,y])
					else:
						h['grid'].append(-1)
			
			h['job'] = copy.deepcopy(self.job)
			#relative job location
			h['relative_job'] = [self.ant[i][0]-self.job[0], self.ant[i][1]-self.job[1]]
			h['action'] = [0 for _ in range(self.n_actions)]
			
			agents.append(h)

		return [agents, copy.deepcopy(self.job)]

	def render(self, VF=None):
		text_render = True
		if not text_render:
			for i in range(self.n_agents):
				plt.scatter(self.ant[i][0], self.ant[i][1], color = 'blue')
				# Add text to show scores
				plt.text(self.ant[i][0], self.ant[i][1], str(self.su[i]))
			plt.scatter(self.job[0], self.job[1], color = 'red', marker='x')
			plt.axis("off")
			plt.axis("equal")
			# Make gridlines
			for i in range(self.gridsize+1):
				# Horizontal lines
				l = i - 0.5
				plt.plot([-0.5,self.gridsize-0.5],[l,l], color='black')
				# Vertical lines
				plt.plot([l,l],[-0.5,self.gridsize-0.5], color='black')	
				
			plt.xlim(-1 , self.gridsize)
			plt.ylim(-1 , self.gridsize)
			plt.ion()
			plt.pause(0.01)
			plt.close()
		else:
			## Text rendering
			# Print a * for each agent, and a # for the job. Each grid location is 3 spaces
			pstr = ""
			for i in range(self.gridsize):
				for j in range(self.gridsize):
					if self.grid[i,j]!=0:
						if self.job[0]==i and self.job[1]==j:
							# print("#*", end="  ")
							pstr += "#*  "
						else:
							# print("*", end="   ")
							pstr += "*   "
					elif self.job[0]==i and self.job[1]==j:
						# print("#", end="   ")
						pstr += "#   "
					else:
						# print(".", end="   ")
						pstr += ".   "
				# print()
				pstr += "\n"
			# print("Score", self.su)
			pstr += "Score: " + str(self.su) + "\n\n"
			print(pstr)


	def get_post_decision_state_agent(self, state, action, ind):
		# For a single agent
		s_i = copy.deepcopy(state)
		obs_template = self.get_observation_template()
		if action!=-1:
			# new_x, new_y = self.pre_move(s_i[:2], action)
			# #apply action
			# s_i[0] = new_x 
			# s_i[1] = new_y
			# # # update the action
			# shift locations
			loc_feats = ['loc', 'job', 'relative_job']
			for feat in loc_feats:
				if feat in obs_template:
					st_ind = obs_template[feat][0]
					x,y = s_i[st_ind:st_ind+2]
					new_x, new_y = self.pre_move([x,y], action)
					s_i[st_ind] = new_x
					s_i[st_ind+1] = new_y
			# update the grid
			if 'grid' in obs_template:
				grid_ind = obs_template['grid'][0]
				grid_end = obs_template['grid'][1]
				grid_len = grid_end - grid_ind
				grid = s_i[grid_ind:grid_end]
				
				shiftx, shifty = self.pre_move([0,0], action)
				# shift the grid. Pad with 0s for the unknown locations
				new_grid = np.zeros(grid_len)
				# reshape the grid to a 2D array
				grid_side = int(np.sqrt(grid_len))
				grid = np.array(grid).reshape((grid_side, grid_side))
				# Add a padding layer of 0s
				grid = np.pad(grid, 1, mode='constant', constant_values=0)
				# select the shifted grid
				new_grid = grid[1 + shiftx:1 + shiftx + grid_side, 1 + shifty:1 + shifty + grid_side]
				s_i[grid_ind:grid_ind+grid_len] = new_grid.flatten()
		return s_i

	def get_valid_locations(self, envt):
		# Get the valid locations for each agent
		# obs is the observation of the environment
		# Returns a mapping of each action to a valid location
		valid_locs = []
		for i in range(self.n_agents):
			h = envt.ant[i]
			valid = {}
			for act in range(5):
				new_x, new_y = self.pre_move(h[:2], act)
				if 0<=new_x<self.gridsize and 0<=new_y<self.gridsize:
					# If the move is legal, add it to the valid locations
					valid[act] = self.map_grid_to_idx(new_x, new_y)
				else:
					# valid[act] = -1
					valid[act] = self.map_grid_to_idx(h[0], h[1])
			valid_locs.append(valid)
		return valid_locs

	def get_resource_counts(self, envt):
		n_locs = envt.gridsize**2
		return [1 for _ in range(n_locs)]
	
	def get_agent_constraints(self, envt):
		# Locations with agents are not valid
		# Add agent constraints: list of locations of other agents
		agent_constraints = []
		for i in range(self.n_agents):
			illegal_locs = [self.map_grid_to_idx(envt.ant[j][0], envt.ant[j][1]) for j in range(self.n_agents) if j!=i]
			agent_constraints.append(illegal_locs)
		return agent_constraints

	def compute_best_actions(self, model, envt, obs, epsilon=0, beta=0, direction='both', use_greedy=False):
		n_agents = envt.n_agents
		n_actions = self.n_actions

		#  Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand() for _ in range(n_actions)] for _ in range(n_agents)]
			return self.compute_allocation(Qvals, envt)
		
		# Greedy strategy
		if use_greedy:
			Qvals = [[-1000000 for _ in range(n_actions)] for _ in range(n_agents)]
			for i in range(n_agents):
				for j in range(n_actions):
					new_loc = self.pre_move(envt.ant[i], j)
					Qvals[i][j] = 5 - get_distance(new_loc, envt.job)

		else:
			# Get Q values from the model
			Qvals = model.get_QValues(self, obs[0])

		return self.compute_allocation(Qvals, envt)

	def compute_allocation(self, Qvals, envt):
		# Convert Qvals to n_agens x n_locations for ILP matching
		n_agents = envt.n_agents
		n_actions = self.n_actions
		n_locs = envt.gridsize**2
		valid_locs = envt.get_valid_locations(envt)

		Qvals_loc = [[-1000000 for _ in range(n_locs)] for _ in range(n_agents)]
		for i in range(n_agents):
			for act in range(n_actions):
				if act in valid_locs[i]:
					Qvals_loc[i][valid_locs[i][act]] = Qvals[i][act]
		resource_counts = self.get_resource_counts(envt)
		agent_constraints = self.get_agent_constraints(envt)

		locations = get_assignment(Qvals_loc, resource_counts, agent_constraints)

		# Convert locations to actions
		actions = [-1]*n_agents
		for i in range(n_agents):
			for act in valid_locs[i]:
				if valid_locs[i][act]==locations[i]:
					actions[i] = act
					break 
		if -1 in actions:
			print("Invalid action")
			print(actions)
		return actions
	
	def compute_neighbors(self):
		distances = distance_matrix(self.ant, self.ant, p=float('+inf'))
		distances = np.array(distances).astype(np.float64)
		for i in range(self.n_agents):
			distances[i,i] = float('+inf')
		distances = distances.argsort()[:,:self.n_neighbors]
		self.neighbors_dist_order = distances
		# self.neighbors_nearby = [[i for i in range(self.n_neighbors)] for _ in range(self.n_agent)]
		self.neighbors_nearby = [[] for _ in range(self.n_agents)]
		for k in range(self.n_agents):
			index = 0
			for i in range(-1, 2):
				for j in range(-1, 2):
					if i!=0 or j!=0:
						if 0<=self.ant[k][0]+i<self.gridsize and 0<=self.ant[k][1]+j<self.gridsize:
							# if there is an agent there, add it to the neighbors
							if self.grid[self.ant[k][0]+i, self.ant[k][1]+j]!=0:
								self.neighbors_nearby[k].append(index)
						index+=1
	
	def get_illegal_actions(self, idx, actions):
		# TODO: ensure this doesn't result in agents moving into each other
		# Needs a check in step()
		# get locations of other agents
		illegal_locations = []
		for i in range(self.n_agents):
			if i!=idx:
				#append the location of the agent
				illegal_locations.append(self.map_grid_to_idx(self.ant[i][0], self.ant[i][1]))
				# compute the grid loc of the agent after action
				if actions[i]!=None:
					new_x, new_y = self.pre_move(self.ant[i], actions[i])
					illegal_locations.append(self.map_grid_to_idx(new_x, new_y))
		illegal_actions = []
		for i in range(self.n_actions):
			new_x, new_y = self.pre_move(self.ant[idx], i)
			if self.map_grid_to_idx(new_x, new_y) in illegal_locations:
				illegal_actions.append(i)
		return illegal_actions


class JobAllocationEnvt(DECAEnvt):
	'''
	The allocation version of the Job environment
	Many workers that desire to work on a job. As long as worker occupies the job's location, they get a reward
	Central agent approach.
	Actions are to take the job or leave it. Only one agent can choose to take it.
	If an agent is given the job, the resource is locked until they get it once.
	If an agent is unable to get out of the way, they still get the reward, but half?
	'''
	def __init__(self, 
		  n_agents, 
		  reallocate=True, 
		  simple_obs=False, 
		  warm_start=0,
		  past_discount=0.995,
		  ):
		
		super().__init__(warm_start, past_discount)
		
		self.n_agents = n_agents
		self.n_actions = 2
		self.n_neighbors = min(3, n_agents-1)
		self.neighbors_size = self.n_neighbors
		
		self.reset()

		self.reallocate = reallocate
		self.simple_obs = simple_obs
		self.state_variables = ['available', 'su', 'discounted_su', 'occupied']

		self.observation_space = ['discounted_su', 'available', 'occupied']
	
	def reset(self):
		# No grid. Each agent is allowed to take the job or not
		# If the agent takes the job, they get a reward of 1
		# If the resource is occupied, another agent cannot take the job

		self.available = True
		self.occupied = [False]*self.n_agents

		self.job_reward = 1

		self.su = np.zeros(self.n_agents)
		w = self.warm_start/4
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2
			for _ in range(self.n_agents)])		
		
	# Each action is one of [take, leave]
	def step(self, actions):
		assert sum(actions)<=1, "Only one agent can take the job"
		# actions[i] is the action taken by agent i
		re = [0]*self.n_agents  #rewards. If an agent is on the job, get reward of 1
		for i in range(self.n_agents):
			assert actions[i] in [0,1], "Invalid action"
			if actions[i]==0:
				self.occupied[i] = False
			else:
				self.occupied[i] = True
				self.available = False
				re[i] = self.job_reward
		
		# check if the job is available
		if any(self.occupied):
			self.available = False
		else:
			self.available = True
			
		self.su+=np.array(re)
		#Update the discounted utilities
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)
		# print('Occupied', self.occupied)
		# print("Available", self.available)
		# print(actions)
		return re

	def get_stateful_observation(self):
		agents = []
		for i in range(self.n_agents):
			h={}
			h['util'] = self.su[i]
			mean_disc_su = np.mean(self.discounted_su)
			if mean_disc_su==0:
				h['relative_util'] = 0
			else:
				h['relative_util'] = self.discounted_su[i]/mean_disc_su - 1
			h['discounted_su'] = self.discounted_su[i]
			h['action'] = [0 for _ in range(self.n_actions)]
			h['available'] = 1 if self.available else 0
			h['occupied'] = 1 if self.occupied[i] else 0
			agents.append(h)

		return [agents, self.available]

	def render(self, VF=None):
		text_render = True
		# plot as a cross. Move the busy agent to the center
		if not text_render:
			for i in range(self.n_agents):
				if self.occupied[i]:
					plt.scatter(0, 0, color = 'blue')
				else:
					rotation = (np.pi/self.n_agents)*i
					x = np.cos(rotation)
					y = np.sin(rotation)
					plt.scatter(x, y, color = 'blue')
			plt.axis("off")
			plt.axis("equal")
			plt.xlim(-1.5 , 1.5)
			plt.ylim(-1.5 , 1.5)
			plt.ion()
			plt.pause(0.01)
			plt.close()
		else:
			## print each agent's utility, and a * next to the agent that has the job
			pstr = ""
			for i in range(self.n_agents):
				pstr += str(round(self.su[i],2))
				if self.occupied[i]:
					pstr += "*"
				else:
					pstr += "."
				pstr += "   "
			pstr += "\n"
			print(pstr)


	def get_post_decision_state_agent(self, state, action, ind):
		# For a single agent
		s_i = copy.deepcopy(state)
		obs_template = self.get_observation_template()
		if action!=-1:
			if 'assigned' in obs_template:
				ind = obs_template['assigned'][0]
				if action==1:
					s_i[ind] = 1
			if 'occupied' in obs_template:
				ind = obs_template['occupied'][0]
				if action==0:
					s_i[ind] = 0
				else:
					s_i[ind] = 1
			if 'action' in obs_template:
				ind = obs_template['action'][0]
				s_i[ind+action] = 1
		return s_i

	def get_resource_counts(self, envt):
		resource_counts = [self.n_agents, 1]
		return resource_counts

	def get_agent_constraints(self, envt):
		# Only the agent occupying the job can take it, others cannot
		agent_constraints = []
		for i in range(self.n_agents):
			if self.occupied[i]:
				agent_constraints.append([])
			else:
				if self.available:
					agent_constraints.append([])
				else:
					agent_constraints.append([1])
		return agent_constraints

	def compute_best_actions(self, model, envt, obs, epsilon=0, beta=0, direction='both', use_greedy=False):
		n_agents = envt.n_agents
		n_actions = self.n_actions

		#  Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand() for _ in range(n_actions)] for _ in range(n_agents)]
			return self.compute_allocation(Qvals, envt)
		
		# Greedy strategy
		if use_greedy:
			Qvals = [[-1000000 for _ in range(n_actions)] for _ in range(n_agents)]
			for i in range(n_agents):
				for j in range(n_actions):
					new_loc = self.pre_move(envt.ant[i], j)
					Qvals[i][j] = 5 - get_distance(new_loc, envt.job)

		else:
			# Get Q values from the model
			Qvals = model.get_QValues(self, obs[0])

		return self.compute_allocation(Qvals, envt)
	
	def compute_neighbors(self):
		pseudo_distances = np.zeros((self.n_agents, self.n_agents))
		for i in range(self.n_agents):
			pseudo_distances[i, i] = float('+inf')
		self.neighbors_dist_order = pseudo_distances.argsort()[:,:self.n_neighbors]
		self.neighbors_nearby = [[i for i in range(self.n_neighbors)] for j in range(self.n_agents)]
	
	def get_illegal_actions(self, idx, actions):
		if not self.available:
			# can't take the job if it is already taken, except if the agent is already on the job
			return [1] if not self.occupied[idx] else []

		illegal_actions = []
		for act in actions:
			if act==1:
				illegal_actions.append(1)
		return illegal_actions


class PlantEnvt(DECAEnvt):
	"""
	The resource allocation version of the environment. Each agent describes preferences over different resources, and the allocation matches them to one.
	"""
	def __init__(self, 
			n_agents=5,
			gridsize=12,
			n_resources=8,
			reallocate=True, 
			simple_obs=False, 
			warm_start=0,
			past_discount=0.995,
			):
		"""
		3 types of resources
		Agents each have unique requirements for combinations of resources
		Once the requirements are satisfied, agents produce one 'unit' and receive a reward, consuming the resource.
		Allocate resources.
		"""
		super().__init__(warm_start, past_discount)
		self.n_agents = n_agents
		self.gridsize = gridsize
		self.n_resources = n_resources
		self.n_actions = n_resources+1
		self.n_neighbors = 4
		self.neighbors_size = 24
		self.simple_obs = simple_obs
		self.warm_start = warm_start

		self.state_variables = ['grid', 'ant', 'posessions', 'resources', 'resource_types', 'requirements', 'su', 'discounted_su', 'targets']
		self.observation_space = ['loc', 'relative_util', 'posessions', 'requirements', 'resources']
		self.observation_space = ['relative_util', 'requirements', 'posessions', 'target'] # For decaf
		self.observation_space = ['relative_util', 'requirements', 'posessions', 'resource_distances', 'target'] # For SOTO
		self.observation_space = ['relative_util', 'requirements', 'posessions', 'can_take_resource', 'resource_distances', 'target'] # For SOTO
		self.observation_space = ['relative_util', 'grid', 'requirements', 'posessions', 'can_take_resource', 'resource_distances', 'target'] # For + testing
		# TODO: Have tried the above without dxdy for resource dists, and without target.
		# self.observation_space = ['relative_util', 'requirements', 'posessions', 'can_take_resource', 'resource_distances', 'target', 'action'] # For action_id
		# self.observation_space = ['loc', 'relative_util', 'requirements', 'posessions', 'grid'] # For SOTO - very hard to learn from, even for decaf
		
		self.reset()
	
	def reset(self):
		# The grid is a 2D array of size gridsize x gridsize
		# Each agent is assigned a random location on the grid
		# Empty grid locations are 0
		# Agent locations are not displayed. Agents can move through each other
		# Resource locations are 1,2,3. There can be multiple resources in one location, but only one is shown

		self.grid = np.zeros((self.gridsize, self.gridsize))
		ant = []
		#Random start locations
		for i in range(self.n_agents):
			loc = np.random.randint(0,self.gridsize,2)
			while self.grid[loc[0],loc[1]]!=0:
				loc = np.random.randint(0,self.gridsize,2)
			ant.append(loc)
		ant = np.array(ant)
		# generate resources
		resources = []
		resource_types = ([0,1,2]*20)[:self.n_resources]
		# shuffle the resource types
		# np.random.shuffle(resource_types)
		for i in range(self.n_resources):
			loc = np.random.randint(1,self.gridsize-1,2)
			resources.append(loc)
			# resource_types.append(np.random.randint(3))
			self.grid[loc[0],loc[1]]=resource_types[i]+1
		
		# requirements=[[2, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 1, 2]]
		requirements=[[2, 1, 0], [1, 0, 1], [1, 0, 0], [1, 3, 0], [0, 1, 2]] # One agent can get units for really cheap, one agent really expensive. Also increase number of steps.

		self.ant = ant
		self.targets = [None]*self.n_agents # Will be [resource_id, distance, resource_type]
		self.su = np.zeros(self.n_agents)
		self.resources = np.array(resources)
		self.resource_types = np.array(resource_types)
		self.requirements = np.array(requirements)
		self.posessions = np.array([[0,0,0] for _ in range(self.n_agents)])

		w = 5 #width of the warm start distribution
		w = self.warm_start/4
		self.discounted_su = np.array([
			self.warm_start + np.random.rand()*w - w/2
			for _ in range(self.n_agents)])
		
	def get_move(self, ant, target):
		#Move agent i in direction of target
		#dir is one of [up, down, left, right]
		x,y = ant
		dirx = target[0] - ant[0]
		diry = target[1] - ant[1]
		deltax, deltay = 0,0
		if dirx!=0:
			deltax = dirx//abs(dirx) if dirx!=0 else 0
		else:
			deltay = diry//abs(diry) if diry!=0 else 0
		return x+deltax, y+deltay

	# n_actions = 1 + n_resources
	def step(self, actions):
		res_ids = [a-1 for a in actions]
		# actions[i] is the action taken by agent i
		# -1 means do nothing
		re = [0.]*self.n_agents  #rewards. If an agent constructs a unit, get a reward of one.
		# Shaping reward 
		re_shaping = np.array([0.]*self.n_agents)
		newgrid = np.zeros((self.gridsize, self.gridsize))
		consumed_resources = []
		# Check that an agent already having a target doesn't get another target
		for i in range(self.n_agents):
			if res_ids[i]!=-1 and self.targets[i]!=None:
				print("Invalid action. This is unsafe, aborting")
				print(f"Agent {i} already has a target {self.targets[i]} and is trying to get another {res_ids[i]}")
				print("Aborting")
				exit()
		for i in range(self.n_agents):
			if res_ids[i]!=-1:
				self.targets[i] = [res_ids[i]]
				distance = get_manhattan_distance(self.ant[i], self.resources[res_ids[i]])
				self.targets[i].append(distance)
				# re_shaping[i] += 0.3*self.past_discount**distance
				r_type = self.resource_types[res_ids[i]]
				self.targets[i].append(r_type)
			
			# If the agent has a target, move towards it
			if self.targets[i]!=None:
				loc = self.ant[i]
				target_loc = self.resources[self.targets[i][0]]
				new_x, new_y = self.get_move(loc, target_loc)

				self.ant[i][0] = new_x
				self.ant[i][1] = new_y

				#Check if agent picked up its target resource
				
				# if self.resources[][0]==new_x and self.resources[j][1]==new_y:
				if new_x==target_loc[0] and new_y==target_loc[1]:
					resource_id = self.targets[i][0]
					consumed_resources.append(resource_id)
					r_type = self.resource_types[resource_id]
					self.posessions[i][r_type]+=1

					#reset target
					self.targets[i] = None

					if self.posessions[i][r_type]<=self.requirements[i][r_type]:
						re_shaping[i]+=0.3

		#replenish consumed resources
		for j in consumed_resources:
			self.resources[j] = np.random.randint(1,self.gridsize-1,2)
			# self.resource_types[j] = np.random.randint(3) #Not changing resouce type.
		for j in range(self.n_resources):
			newgrid[self.resources[j][0],self.resources[j][1]]=self.resource_types[j]+1
		# update the grid
		self.grid = newgrid

		# Construct units
		for i in range(self.n_agents):
			n_units = 10000
			for j in range(3):
				if self.requirements[i][j]==0:
					continue
				else:
					count = int(self.posessions[i][j]/self.requirements[i][j])
					if count<n_units:
						n_units = count
			re[i]+=n_units
			for j in range(3):
				self.posessions[i][j]-=self.requirements[i][j]*n_units

		self.su+=np.array(re)
		re_return = np.array(re) + re_shaping
		#Update the discounted utilities
		self.discounted_su = self.discounted_su*self.past_discount + np.array(re)

		return re_return

	def get_closest_resources_by_type(self, agent, resources, resource_types):
		# returns the relative position of the closest resource of each type
		relative_closest_resource = [[-100,-100],[-100,-100],[-100,-100]]
		closest_distances = [10000, 10000, 10000]
		for j in range(len(resources)):
			r_type = resource_types[j]
			dist = get_distance(agent, resources[j])
			if dist<closest_distances[r_type]:
				closest_distances[r_type] = dist
				relative_closest_resource[r_type] = [agent[0] - resources[j][0], agent[1] - resources[j][1]]
		return relative_closest_resource, closest_distances
	
	def get_surrounding_grid(self, loc, grid, size=5):
		# Get the 3x3 grid around the location
		x, y = loc
		grid_3x3 = []
		# for i in range(x-1, x+2):
		# 	for j in range(y-1, y+2):
		for i in range(x-(size-1)//2, x+(size-1)//2+1):
			for j in range(y-(size-1)//2, y+(size-1)//2+1):
				if 0<=i<self.gridsize and 0<=j<self.gridsize:
					grid_3x3.append(grid[i,j])
				else:
					grid_3x3.append(-1)
		return grid_3x3

	def get_stateful_observation(self):
		agents = []
		for i in range(self.n_agents):
			closest_resource, distances = self.get_closest_resources_by_type(self.ant[i], self.resources, self.resource_types)
			closest_locs = [coord for loc in closest_resource for coord in loc]
			all_resource_dists = []
			for j in range(self.n_resources):
				dx, dy = self.ant[i][0] - self.resources[j][0], self.ant[i][1] - self.resources[j][1]
				all_resource_dists.append(dx)
				all_resource_dists.append(dy)
				all_resource_dists.append(get_distance(self.ant[i], self.resources[j]))
				all_resource_dists.append(self.resource_types[j])
			if self.targets[i]!=None:
				tx = self.resources[self.targets[i][0]][0] - self.ant[i][0]
				ty = self.resources[self.targets[i][0]][1] - self.ant[i][1]
				t = [tx, ty, self.targets[i][1], self.targets[i][2]]
			else:
				t = [-100,-100,100, -1] # dx,dy,dist,type
			mean_disc_su = np.mean(self.discounted_su)
			if mean_disc_su==0:
				rel_util = 0
			else:
				rel_util = self.discounted_su[i]/mean_disc_su - 1
			h = {
				"loc":[self.ant[i][0], self.ant[i][1]],
				"util":self.su[i],
				"relative_util":rel_util,
				"posessions":self.posessions[i],
				"requirements":self.requirements[i],
				"needs":[max(r-p,0) for r,p in zip(self.requirements[i], self.posessions[i])],
				"resources":closest_locs,
				"resource_distances":all_resource_dists,
				'can_take_resource':int(self.targets[i]==None),
				'target':t,
				'action':[0 for _ in range(self.n_actions)]
			}

			#Get info about other agents
			others = []
			# append locations of all other agents and their needs and targets
			for j in range(self.n_agents):
				reqs = self.requirements[j]
				poss = self.posessions[j]
				need = [r-p for r,p in zip(reqs, poss)]
				if j!=i:
					others.append([
						self.ant[j][0],self.ant[j][1],
						self.su[j],
						need[0], need[1], need[2],
						self.targets[j][0] if self.targets[j]!=None else -1,
					])
			#flatten
			others = [feature for other in others for feature in other]
			h['other_agents'] = others
			

			# alt: get the 3x3 grid around the agent
			h['grid'] = self.get_surrounding_grid(self.ant[i], self.grid)

			agents.append(h)

		return [agents, copy.deepcopy(self.resources)]

	
	def render(self):
		fig, ax = plt.subplots()
		ax.set_xlim(0, self.gridsize)
		ax.set_ylim(0, self.gridsize)
		
		# Draw resources
		for i in range(self.n_resources):
			x, y = self.resources[i]
			color = ['red', 'green', 'blue'][self.resource_types[i]]
			ax.add_patch(plt.Circle((x + 0.5, y + 0.5), 0.4, color=color))
		
		# Draw agents
		agent_colors = plt.cm.viridis(np.linspace(0, 1, self.n_agents))
			
		for i in range(self.n_agents):
			x, y = self.ant[i]
			ax.add_patch(plt.Rectangle((x + 0.25, y + 0.25), 0.5, 0.5, color=agent_colors[i]))
		
		# Draw grid
		for i in range(self.gridsize+1):
			plt.plot([i,i], [0,self.gridsize], color='black')
			plt.plot([0,self.gridsize], [i,i], color='black')
		
		# Draw text
		for i in range(self.n_agents):
			ax.text(self.ant[i][0]+0.5, self.ant[i][1]+0.5, str(self.su[i]), ha='center', va='center')
		
		plt.ion()
		plt.pause(0.01)
		plt.close()
		return


	def get_post_decision_state_agent(self, state, action, idx):
		s_i = copy.deepcopy(state)
		res_id = action - 1
		if self.observation_template == None:
			self.get_observation_template()
		# Need to also correct the location of the resources
		if res_id!=-1:
			old_loc = self.ant[idx]
			new_x, new_y = self.get_move(old_loc, self.resources[res_id])
			new_loc = [new_x, new_y]
			#apply res_id
			if 'loc' in self.observation_space:
				loc_inds = self.observation_template['loc']
				s_i[loc_inds[0]] = new_x
				s_i[loc_inds[1]] = new_y
			# Change the relative loc of the resources
			if 'resources' in self.observation_space:
				res_inds = self.observation_template['resources']
				closest_resource, distances = self.get_closest_resources_by_type(new_loc, self.resources, self.resource_types)
				# closest_locs = closest_resource[1] # ONLY FOR TEMPORARY TESTING
				closest_locs = [coord for loc in closest_resource for coord in loc]
				for k in range(len(closest_locs)):
					s_i[res_inds[0]+k] = closest_locs[k]
			if 'grid' in self.observation_space:
				grid_inds = self.observation_template['grid']
				#get the 5x5 grid around the agent
				copy_grid = copy.deepcopy(self.grid)
				if 0<=new_x<self.gridsize and 0<=new_y<self.gridsize:
					# copy_grid[new_x, new_y] = 4 
					grid = self.get_surrounding_grid(new_loc, copy_grid)
				else:
					grid = self.get_surrounding_grid(old_loc, copy_grid)
				for k in range(len(grid)):
					s_i[grid_inds[0]+k] = grid[k]
			if 'target' in self.observation_space:
				target_inds = self.observation_template['target']
				target = []
				if res_id!=-1:
					# will always enter
					distance = get_manhattan_distance(new_loc, self.resources[res_id])
					r_type = self.resource_types[res_id]
					dx = self.resources[res_id][0] - new_x
					dy = self.resources[res_id][1] - new_y
					target = [dx, dy, distance, r_type]
				else:
					target = [-100,-100,100, -1]
				for k in range(len(target)):
					s_i[target_inds[0]+k] = target[k]
			if 'resource_distances' in self.observation_space:
				dist_inds = self.observation_template['resource_distances']
				all_resource_dists = []
				for j in range(self.n_resources):
					dx, dy = new_x - self.resources[j][0], new_y - self.resources[j][1]
					all_resource_dists.append(dx)
					all_resource_dists.append(dy)
					all_resource_dists.append(get_distance(new_loc, self.resources[j]))
					all_resource_dists.append(self.resource_types[j])
				for k in range(len(all_resource_dists)):
					s_i[dist_inds[0]+k] = all_resource_dists[k]
			if 'action' in self.observation_space:
				action_inds = self.observation_template['action']
				for k in range(self.n_actions):
					s_i[action_inds[0]+k] = 0
				s_i[action_inds[0]+action] = 1
		return s_i
	
	def get_resource_counts(self, envt):
		n_agents = envt.n_agents
		targets = envt.targets
		occupied_resources = set([targets[j][0] for j in range(n_agents) if targets[j] is not None])
		resource_counts = [n_agents] # First action does not have a resource restriction
		for i in range(envt.n_actions-1):
			# Add available resources
			if i not in occupied_resources:
				resource_counts.append(1)
			else:
				resource_counts.append(0)
		return resource_counts
	
	def get_agent_constraints(self, envt):
		agent_constraints = []
		for i in range(envt.n_agents):
			if envt.targets[i]!=None:
				illegal_resources = [j+1 for j in range(envt.n_resources)]
			else:
				illegal_resources = []
			agent_constraints.append(illegal_resources)
		return agent_constraints

	def compute_best_actions(self, model, envt, obs, epsilon=0, beta=0, direction='both', use_greedy=False):
		n_agents = envt.n_agents
		n_actions = envt.n_actions

		#  Get a random action with probability epsilon
		if np.random.rand()<epsilon:
			Qvals = [[np.random.rand() for _ in range(n_actions)] for _ in range(n_agents)]
			return self.compute_allocation(Qvals, envt)
		
		# Greedy strategy
		if use_greedy:
			Qvals = [[-1000000 for _ in range(n_actions)] for _ in range(n_agents)]
			for i in range(n_agents):
				# find the closest resource that is needed
				needs = [envt.requirements[i][j] - envt.posessions[i][j] for j in range(3)]
				if needs[0]>0:
					target = envt.resources[0]
					Qvals[i][0] = 12 - get_distance(envt.ant[i], target)
				else:
					Qvals[i][0] = -1000000

		else:
			# Get Q values from the model
			Qvals = model.get_QValues(self, obs[0])

		return self.compute_allocation(Qvals, envt)
	
	def compute_neighbors(self):
		distances = distance_matrix(self.ant, self.ant, p=float('+inf'))
		distances = np.array(distances).astype(np.float64)
		for i in range(self.n_agents):
			distances[i,i] = float('+inf')
		distances = distances.argsort()[:,:self.n_neighbors]
		self.neighbors_dist_order = distances

		# To use this, change neighbors_size to n_neighbors!!
		# self.neighbors_nearby = [[i for i in range(self.n_neighbors)] for _ in range(self.n_agents)]

		# in the 5x5 grid around the agent
		self.neighbors_nearby = [[] for _ in range(self.n_agents)]
		for k in range(self.n_agents):
			index = 0
			for i in range(-2, 3):
				for j in range(-2, 3):
					if i!=0 or j!=0:
						locx = self.ant[k][0]+i
						locy = self.ant[k][1]+j
						if 0<=locx<self.gridsize and 0<=locy<self.gridsize:
							# if there is an agent there, add it to the neighbors
							for kk in range(self.n_agents):
								if self.ant[kk][0]==locx and self.ant[kk][1]==locy:
									self.neighbors_nearby[k].append(index)
						index+=1

	def get_illegal_actions(self, idx, actions):
		# get locations of other agents
		illegal_actions = []
		if self.targets[idx]!=None:
			illegal_actions = [i+1 for i in range(self.n_resources)]
		occupied_resources = set([self.targets[j][0] for j in range(self.n_agents) if self.targets[j] is not None])
		# consider just allocated resources
		for act in actions:
			if act!=None and act!=0:
				occupied_resources.add(act-1)
		# print("Occupied resources", occupied_resources)
		for i in range(self.n_actions):
			if i-1 in occupied_resources:
				illegal_actions.append(i)
		# print("Illegal actions", illegal_actions)
		return illegal_actions
