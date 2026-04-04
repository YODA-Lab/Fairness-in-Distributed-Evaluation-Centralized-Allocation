"""
Second version of SOTO implementation
Randomly orders agents and selects actions in order
	masking actions that are not available
	using the policy network to select actions
"""
import os, time  
import numpy as np
import tensorflow as tf
import copy
import csv

from agent_utils import get_agent, take_env_step, training_step, post_episode_hk, get_env, run_validation, save_best_model
from utils import EpsilonDecay, add_epi_metrics_to_logs, add_metric_to_logs, get_metrics_from_rewards
from process_args import process_args
from fairness_functions import fairness_router

# For importing from SOTO code
from keras.utils import to_categorical
from DFRL.common.utils import eligibility_traces, default_config, make_env, str2bool, get_omega, get_more_obs_com, discount_rewards
from DFRL.common.ppo_independant import PPOPolicyNetworkMasked, ValueNetwork
from collections import deque
# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


save_path = "logs/SOTO/GGF/"
args, train_args = process_args(save_path=save_path)

if args.training and args.logging:
	log_dir = f"{args.save_path}/"
	print("Logging to {} \n\n\n\n".format(log_dir))
	summary_writer = tf.summary.create_file_writer(log_dir)
else:
	summary_writer = None

# Env params
M, M_train, M_val = get_env(args.env_name, args, train_args)
M_val.external_trigger = True

#SOTO params
# LAMBDA = -1 if env is Job else 0.97
LAMBDA = -1 if args.env_name == "Job" else 0.97
GAMMA = 0.98
lr_actor = train_args.learning_rate
# twophase_proportion = 0.2 if args.env_name == "Plant" else 0.5
twophase_proportion = 0.5 # half the episodes are in the first phase


args.fairness_type = 'ggf'
config = {"agent":
		  {
			  "ggi_type":"xpowerminusn", 
			  "twophase_proportion":twophase_proportion, 
			  "lambda":LAMBDA, 
			  "lr_actor":lr_actor,
			  "ggi_constant":2
			}
		}

n_agent = M.n_agents
n_episode = args.n_episode
omega = get_omega(config, n_agent)
fairness_function = fairness_router(args.fairness_type, weights=omega)

obs = M.get_obs()
num_features = len(obs[0][0])

T = 25 if args.env_name == "Job" else 50
gPi = []
Pi = []
gV = []
V = []
advantage_obs_size = n_agent
use_neighbors = True
more_obs_size = 0
if use_neighbors:
	more_obs_size = M.neighbors_size+1 #TODO: Include this as well, and compare with DECAF in a similar setup
print('More obs size:', more_obs_size)
SO_obs_size = num_features
TO_obs_size = num_features + more_obs_size + advantage_obs_size + M.n_actions

shared_nets = True
#lets try sharing the network
if shared_nets:
	gPiShared = PPOPolicyNetworkMasked(num_features=SO_obs_size, num_actions=M.n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor)
	PiShared = PPOPolicyNetworkMasked(num_features=TO_obs_size , num_actions=M.n_actions, layer_size=64, epsilon=0.1, learning_rate=lr_actor)
	gVShared = ValueNetwork(num_features=SO_obs_size, hidden_size=256, learning_rate=0.001)
	VShared = ValueNetwork(num_features=TO_obs_size, hidden_size=256, learning_rate=0.001)
	for i in range(n_agent):
		gPi.append(gPiShared)
		Pi.append(PiShared)
		gV.append(gVShared)
		V.append(VShared)
else:
	for i in range(n_agent):
		gPi.append(PPOPolicyNetworkMasked(num_features=SO_obs_size, num_actions=M.n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor))
		Pi.append(PPOPolicyNetworkMasked(num_features=TO_obs_size , num_actions=M.n_actions, layer_size=64, epsilon=0.1, learning_rate=lr_actor))
		gV.append(ValueNetwork(num_features=SO_obs_size, hidden_size=256, learning_rate=0.001))
		V.append(ValueNetwork(num_features=TO_obs_size, hidden_size=256, learning_rate=0.001))


memory_ep_rewards = [deque() for _ in range(n_agent)]
average_jpi = np.zeros(n_agent)
episode_util_logs = []

#debug logging file. json-like saving each episode data incrementally.
import json
def convert_to_serializable(obj):
	"""
	Recursively converts NumPy data types to native Python data types.
	"""
	if isinstance(obj, np.integer):
		return int(obj)
	elif isinstance(obj, np.floating):
		return float(obj)
	elif isinstance(obj, np.ndarray):
		return obj.tolist()  # Convert NumPy arrays to lists
	elif isinstance(obj, dict):
		return {key: convert_to_serializable(value) for key, value in obj.items()}
	elif isinstance(obj, list):
		return [convert_to_serializable(element) for element in obj]
	else:
		return obj
debug = False
if debug:
	write_file = f"debug_logs_{args.env_name}_mask2.json"
	print("Writing debug logs to", write_file)
	with open(write_file, "w") as f:
		f.write("[\n")

run_metrics = {'system_utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
# while i_episode<args.n_episode:
for i_episode in range(1, args.n_episode+1):
	beta = max(1 - float(i_episode) / (twophase_proportion * float(n_episode)), 0.0)

	memory_ep_rewards = [deque() for _ in range(n_agent)]
	average_jpi = np.zeros(n_agent)

	ep_actions = [[] for _ in range(n_agent)]
	ep_rewards = [[] for _ in range(n_agent)]
	ep_states  = [[] for _ in range(n_agent)]
	ep_f_rewards = [[] for _ in range(n_agent)]
	ep_masks = [[] for _ in range(n_agent)]

	greedy = np.zeros(n_agent).astype(bool)
	for i in range(n_agent):
		greedyc = np.random.rand() <= beta
		greedy[i] = greedyc

	score=0  #Central agent score = sum of agent scores
	su = [0.] * n_agent
	su = np.array(su)

	M.reset()
	obs = M.get_obs()
	if use_neighbors:
		neighbors = M.neighbors()

	nancheck = 1
	logstep = 100

	# # pick a random agent order
	# agent_order = np.random.permutation(n_agent) #TODO: Decide where this goes.
	# Run the episode
	for steps in range(1, args.max_steps+1):
		step_log = {"agents":{}} #log for each step
		#For each agent, select the action 
		actions = [None for _ in range(n_agent)]
		action_probs = [None for _ in range(n_agent)]
		# pick a random agent order
		agent_order = np.random.permutation(n_agent)
		# #reverse order, from n to 0
		# agent_order = np.arange(n_agent-1, -1, -1)
		
		for i in agent_order:
			h = copy.deepcopy(obs[0][i])
			illegal_actions = M.get_illegal_actions(i, actions)
			# mask the actions
			mask = np.ones(M.n_actions)
			mask[illegal_actions] = 0
			if not greedy[i]:
				#Only mask the final action? Maybe not
				more_obs = gPi[i].get_dist(np.array([h]), mask=np.array([mask]))[0]
				# more_obs = gPi[i].get_dist(np.array([h]), mask=np.array([mask]))[0]
				h.extend(more_obs)
				if use_neighbors:
					more_obs = get_more_obs_com(True, neighbors, average_jpi, i, more_obs_size)
					h.extend(more_obs)
				# Advantages signal
				more_obs = average_jpi
				more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
				h.extend(more_obs)
				p = Pi[i].get_dist(np.array([h]), mask=np.array([mask]))[0]
			else:
				p = gPi[i].get_dist(np.array([h]), mask=np.array([mask]))[0]
			
			ep_states[i].append(h)
			# decay the probability of the 0 action, if other actions are available
			# decay_factor = 0.95**i_episode
			# if np.sum(mask) > 1:
			# 	p[0] = p[0] * (1-decay_factor)
			# 	p = p / np.sum(p)
			if np.sum(p) != 1:
				p = np.array(p)
				p = p / np.sum(p)
			if np.isnan(p).any():
				print("NAN in probs", p)
				#replace nan with 1/n, re normalize
				p[np.isnan(p)] = 1/len(p)
				if nancheck:
					exit()
				p = p / np.sum(p)

			actions[i] = np.random.choice(M.n_actions, p=p)
			action_probs[i] = p
			if debug and i_episode == 100:
				print("Agent", i, "\nAction", actions[i], "\nProbs", p, "\nMask", mask)
			# print("Agent", i, "\nAction", actions[i], "\nProbs", p, "\nMask", mask)
			# occupied = list(set([M.targets[j][0] for j in range(M.n_agents) if M.targets[j] is not None]))
			# print([o+1 for o in occupied])

			step_log['agents'][int(i)] = {'state':h, 'action':actions[i], 'probs':p, 'mask':mask}
			ep_masks[i].append(mask)
		if debug:
			step_log['system'] = {'step':steps, 'episode':i_episode, 'pre_actions':actions}
			if args.env_name != "BiasedDM":
				occupied = list(set([M.targets[j][0] for j in range(M.n_agents) if M.targets[j] is not None]))
				step_log['system']['occupied'] = [o+1 for o in occupied]
				step_log['system']['targets'] =  M.targets
			if args.env_name == 'Plant':
				step_log['system'][ 'posessions'] = M.posessions
		# try to resolve any conflicts, and allocate random actions in case of ties
		# actions = M.attempt_allocation(actions, M)
		# actions = M.compute_allocation(action_probs, M )
		# Take the step
		su_prev = copy.deepcopy(M.get_fairness_rewards())
		rewards = M.step(actions)
		obs = M.get_obs()
		if M.fairness_vars=='':
			fairness_rewards = rewards
		else:
			su_post = copy.deepcopy(M.get_fairness_rewards())
			fairness_rewards = [su_post[i] - su_prev[i] for i in range(n_agent)]
			# print("Fairness_rewards", fairness_rewards)
			# print(su_post, su_prev)
		if use_neighbors:
			neighbors = M.neighbors()
		
		su += np.array(rewards)
		score += sum(rewards)

		if debug:
			step_log['system']['post_actions'] = actions
			step_log['system']['rewards'] = rewards
			step_log['system']['fairness_rewards'] = fairness_rewards

			# write the log to the file
			# delete the last "]"
			with open(write_file, "rb+") as f:
				f.seek(-1, os.SEEK_END)
				f.truncate()
			with open(write_file, "a") as f:
				#skip for the very first step
				if i_episode>1 or steps>1:
					f.write(",\n")
				else:
					f.write("\n")
				step_log = convert_to_serializable(step_log)
				json.dump(step_log, f)
				f.write("]")
		
		for i in range(n_agent):
			ep_actions[i].append(to_categorical(actions[i], M.n_actions))
			ep_rewards[i].append(rewards[i])
			ep_f_rewards[i].append(fairness_rewards[i])
			# memory_ep_rewards[i].append(rewards[i])
			# average_jpi[i] += rewards[i]
			# Use the fairness rewards for ranking
			memory_ep_rewards[i].append(fairness_rewards[i])
			average_jpi[i] += fairness_rewards[i]
			# if len(memory_ep_rewards[i]) > args.max_steps * 5:
			# 	average_jpi[i] -= memory_ep_rewards[i].popleft()

		if steps % T == 0:
			all_ep_advantages=[]
			for i in range(n_agent):
				ep_actions[i] = np.array(ep_actions[i])
				ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
				ep_f_rewards[i] = np.array(ep_f_rewards[i], dtype=np.float_)
				ep_states[i] = np.array(ep_states[i])
				ep_masks[i] = np.array(ep_masks[i])
				dummy_actions = [None for _ in range(n_agent)]

				if LAMBDA < -0.1:
					if not greedy[i]:
						targets = discount_rewards(ep_f_rewards[i], GAMMA)
						V[i].update(ep_states[i], targets)
						vs = V[i].get(ep_states[i])
					else:
						targets = discount_rewards(ep_rewards[i], GAMMA)
						gV[i].update(ep_states[i], targets)
						vs = gV[i].get(ep_states[i])
				else:
					next_s = copy.deepcopy(obs[0][i])
					if not greedy[i]:
						vs = V[i].get(ep_states[i])
						# illegal_actions = M.get_illegal_actions(i, dummy_actions)
						# mask = np.ones(M.n_actions)
						# mask[illegal_actions] = 0
						# more_obs = gPi[i].get_dist(np.array([obs[0][i]]) , mask=np.array([mask]))[0]
						more_obs = gPi[i].get_dist(np.array([obs[0][i]]))[0]
						next_s.extend(more_obs)
						if use_neighbors:
							more_obs = get_more_obs_com(True, neighbors, average_jpi, i, more_obs_size)
							next_s.extend(more_obs)
						more_obs = average_jpi
						more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
						next_s.extend(more_obs)
						targets = eligibility_traces(ep_f_rewards[i], vs, V[i].get([next_s]), GAMMA, LAMBDA)
						V[i].update(ep_states[i], targets)
					else:
						vs = gV[i].get(ep_states[i])
						targets = eligibility_traces(ep_rewards[i], vs, gV[i].get([next_s]), GAMMA, LAMBDA)
						gV[i].update(ep_states[i], targets)

				ep_advantages = targets - vs
				ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
				all_ep_advantages.append(ep_advantages)

			all_ep_advantages = np.array(all_ep_advantages)
			all_ep_advantages_saved = all_ep_advantages
			sorted_index = average_jpi.argsort()
			sorted_index = [np.where(sorted_index == i)[0][0] for i in range(n_agent)]
			all_ep_advantages = omega[sorted_index] @ all_ep_advantages
			for i in range(n_agent):
				if not greedy[i]:
					Pi[i].update(ep_states[i], ep_actions[i], all_ep_advantages, ep_masks[i])
				else:
					gPi[i].update(ep_states[i], ep_actions[i], all_ep_advantages_saved[i], ep_masks[i])

			ep_actions = [[] for _ in range(n_agent)]
			ep_rewards = [[] for _ in range(n_agent)]
			ep_f_rewards = [[] for _ in range(n_agent)]
			ep_states  = [[] for _ in range(n_agent)]
			ep_masks = [[] for _ in range(n_agent)]

			greedy=np.zeros(n_agent).astype(bool)
			for i in range(n_agent):
				greedyc = np.random.rand() <= beta
				greedy[i] = greedyc
		
		if args.render:
			M.render()
			time.sleep(0.1)
	
	# Post episode housekeeping
	print("Rewards", su)
	print("Fairness_rewards", average_jpi)
	episode_util_logs.append(['train', i_episode, M.su, M.get_fairness_rewards(), M.discounted_su])
	epi_metrics = add_epi_metrics_to_logs(summary_writer, M.su, None, beta, i_episode, args.max_steps, verbose=True, prefix="", logging=args.logging, fair_rewards=M.get_fairness_rewards(), fairness_type=args.fairness_type, fairness_function=fairness_function)
	
	for key, value in epi_metrics.items():
		run_metrics[key].append(value)
		#Print the average metrics
		if i_episode%50==0:
			print("Average "+key+": ", np.mean(run_metrics[key]))

# Final round of validation and saving results
if args.training:
	print("Final Validation")
	num_eps = 50
	M_val.external_trigger = True
	val_metrics = {'system_utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}
	fairness_function = fairness_router(args.fairness_type)
	val_util_logs = []
	for val_eps in range(num_eps):
		M_val.reset()
		average_jpi = np.zeros(n_agent)
		obs = M_val.get_obs()
		if use_neighbors:
			neighbors = M_val.neighbors()
		score = 0
		for steps in range(args.max_steps):
			actions = [None for _ in range(n_agent)]
			# pick a random agent order
			agent_order = np.random.permutation(n_agent)
			for i in agent_order:
				h = copy.deepcopy(obs[0][i])
				# mask the actions
				illegal_actions = M_val.get_illegal_actions(i, actions)
				mask = np.ones(M_val.n_actions)
				mask[illegal_actions] = 0

				#never use greedy in validation
				more_obs = gPi[i].get_dist(np.array([h]), mask=np.array([mask]))[0]
				h.extend(more_obs)
				if use_neighbors:
					more_obs = get_more_obs_com(True, neighbors, average_jpi, i, more_obs_size)
					h.extend(more_obs)
				# Advantages signal
				more_obs = average_jpi
				more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
				h.extend(more_obs)
				p = Pi[i].get_dist(np.array([h]), mask=np.array([mask]))[0]
		
				if np.sum(p) != 1 or np.isnan(p).any():
					p = np.array(p)
					p = p / np.sum(p)

				actions[i] = np.random.choice(M_val.n_actions, p=p)

			# # try to resolve any conflicts, and allocate random actions in case of ties
			# actions = M_val.attempt_allocation(actions, M_val)
			# Take the step
			su_prev = copy.deepcopy(M_val.get_fairness_rewards())
			rewards = M_val.step(actions)
			obs = M_val.get_obs()
			if use_neighbors:
				neighbors = M_val.neighbors()
			score += sum(rewards)
			if M_val.fairness_vars=='':
				fairness_rewards = rewards
			else:
				su_post = copy.deepcopy(M_val.get_fairness_rewards())
				fairness_rewards = [su_post[i] - su_prev[i] for i in range(n_agent)]
			
			for i in range(n_agent):
				average_jpi[i] += fairness_rewards[i]

		val_util_logs.append(['val', val_eps, M_val.su, M_val.get_fairness_rewards(), M_val.discounted_su])
		print("Rewards", score)
		print(M_val.su)
		print(M_val.get_fairness_rewards(), "Fair")
		print(M_val.discounted_su, "Fair")

		metrics = get_metrics_from_rewards(M_val.su, args.learning_beta, fair_rewards=M_val.get_fairness_rewards(), fairness_type=args.fairness_type, fairness_function=fairness_function)
		for key, value in metrics.items():
			val_metrics[key].append(value)


	# val_metrics, val_util_logs = run_validation(num_eps, M_val, agent, args, render=args.render) 
	for log in val_util_logs:
		log[1] = i_episode
		episode_util_logs.append(log)
	
	mean_val_metrics = {}
	for key, value in val_metrics.items():
		mean_val_metrics[key] = np.mean(value)
		# Results file is a csv
	# If the file doesn't exist, create it and write the header
	# Create the directory if it doesn't exist
	results_file = f"Results/SOTO/{args.env_name+args.env_name_mod}results.csv"
	# Also save a copy of the results file in the save_path
	results_file2 = f"{args.save_path}/results.csv"
	create=False
	if not os.path.exists("Results/SOTO"):
		os.makedirs("Results/SOTO")
	file_exists = os.path.exists(results_file)

	all_fields = {}
	for key, value in mean_val_metrics.items():
		all_fields[key] = value
	for arg in vars(args):
		all_fields[arg] = getattr(args, arg)
	for arg in vars(train_args):
		all_fields[arg] = getattr(train_args, arg)
	all_fields["observation_space"] = M.observation_space
	
	with open(results_file, "a", newline="") as f:	
		# Add one row to the csv file
		writer = csv.DictWriter(f, fieldnames=all_fields.keys())
		if not file_exists:
			writer.writeheader()
		writer.writerow(all_fields)
	
	# Save a copy of the results file in the save_path
	file_exists2 = os.path.exists(results_file2)
	with open(results_file2, "a", newline="") as f:	
		# Add one row to the csv file
		writer = csv.DictWriter(f, fieldnames=all_fields.keys())
		if not file_exists2:
			writer.writeheader()
		writer.writerow(all_fields)
	
	#Save the util logs
	headers = ['train/val', 'episode', 'system_utility', 'fairness', 'discounted_utility']
	with open(f"{args.save_path}/util_logs.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerow(headers)
		writer.writerows(episode_util_logs)