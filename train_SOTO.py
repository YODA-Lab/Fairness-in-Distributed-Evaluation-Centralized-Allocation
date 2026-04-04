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
from DFRL.common.ppo_independant import PPOPolicyNetwork, ValueNetwork
from collections import deque
# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


save_path = "logs/SOTO-ILP/GGF/"
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
twophase_proportion = 0.2 if args.env_name == "Plant" else 0.5


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
if shared_nets:
	#lets try sharing the network
	gPiShared = PPOPolicyNetwork(num_features=SO_obs_size, num_actions=M.n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor)
	PiShared = PPOPolicyNetwork(num_features=TO_obs_size , num_actions=M.n_actions, layer_size=64, epsilon=0.1, learning_rate=lr_actor)
	gVShared = ValueNetwork(num_features=SO_obs_size, hidden_size=256, learning_rate=0.001)
	VShared = ValueNetwork(num_features=TO_obs_size, hidden_size=256, learning_rate=0.001)
	for i in range(n_agent):
		gPi.append(gPiShared)
		Pi.append(PiShared)
		gV.append(gVShared)
		V.append(VShared)
else:
	for i in range(n_agent):
		gPi.append(PPOPolicyNetwork(num_features=SO_obs_size, num_actions=M.n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor))
		Pi.append(PPOPolicyNetwork(num_features=TO_obs_size , num_actions=M.n_actions, layer_size=64, epsilon=0.1, learning_rate=lr_actor))
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
	write_file = "debug_logs_matthew.json"
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

	# Run the episode
	for steps in range(1, args.max_steps+1):
		step_log = {"agents":{}} #log for each step
		#For each agent, select the action 
		actions = []
		action_probs = []
		for i in range(n_agent):
			h = copy.deepcopy(obs[0][i])
			if not greedy[i]:
				more_obs = gPi[i].get_dist(np.array([h]))[0]
				h.extend(more_obs)
				if use_neighbors:
					more_obs = get_more_obs_com(True, neighbors, average_jpi, i, more_obs_size)
					h.extend(more_obs)
				# Advantages signal
				more_obs = average_jpi
				more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
				h.extend(more_obs)
				p = Pi[i].get_dist(np.array([h]))[0]
			else:
				p = gPi[i].get_dist(np.array([h]))[0]
			ep_states[i].append(h)
			if np.sum(p) != 1:
				p = np.array(p)
				p = p / np.sum(p)
			if np.isnan(p).any():
				print("NAN in probs", p)
				#replace nan with 1/n, re normalize
				p[np.isnan(p)] = 1/len(p)
				if nancheck:
					exit()
					# q = input("Press Enter to continue... or 0 to stop this message")
					# if q == "0":
					# 	nancheck = 0
				p = p / np.sum(p)

			actions.append(np.random.choice(M.n_actions, p=p))
			action_probs.append(p)
			# Use the policy actions to store transition, not the resolved action.
			# ep_actions[i].append(to_categorical(actions[i], M.n_actions))

			step_log['agents'][i] = {'state':h, 'action':actions[i], 'probs':p}
			
			# # if steps==50 and i_episode%10==0:
			if i_episode%logstep==0 and debug:
				#print the probs
				# print("Targets", M.targets)
				# occupied = set([M.targets[j][0] for j in range(M.n_agents) if M.targets[j] is not None])
				print("Agent", i)
				print(actions[i], p[actions[i]])
				print("Probs", p)

		if debug:
			occupied = list(set([M.targets[j][0] for j in range(M.n_agents) if M.targets[j] is not None]))
			step_log['system'] = {'step':steps, 'episode':i_episode, 'pre_actions':actions, 'occupied':occupied, 'targets':M.targets}
			# step_log['system'][ 'posessions'] = M.posessions
			# if steps==50 and i_episode%10==0:
			if i_episode%logstep==0:
				
				busy_agents = [j for j in range(M.n_agents) if M.targets[j] is not None]
				if len(occupied) <3:
					print("Step", steps)
					print("Occupied", occupied)
					print("Busy agents", busy_agents)
					print("Selected", actions)

		# try to resolve any conflicts, and allocate random actions in case of ties
		# actions = M.attempt_allocation(actions, M)
		actions = M.compute_allocation(action_probs, M )
		# actions = M.attempt_allocation(actions, M, debug=i_episode%logstep==0)
		# if steps==50 and i_episode%10==0:
		# Take the step
		su_prev = copy.deepcopy(M.get_fairness_rewards())
		rewards = M.step(actions)
		obs = M.get_obs()
		if M.fairness_vars=='':
			fairness_rewards = rewards
		else:
			su_post = copy.deepcopy(M.get_fairness_rewards())
			fairness_rewards = [su_post[i] - su_prev[i] for i in range(n_agent)]
		if use_neighbors:
			neighbors = M.neighbors()
		
		if debug:
			if i_episode%logstep==0:
				if len(occupied) <3:
					print("Final   ", actions)
			step_log['system']['post_actions'] = actions
			step_log['system']['rewards'] = rewards
			step_log['system']['fairness_rewards'] = fairness_rewards

			# write the log to the file
			# delete the last "]"
			with open(write_file, "rb+") as f:
				f.seek(-1, os.SEEK_END)
				f.truncate()
			with open(write_file, "a") as f:
				f.write(",\n")
				step_log = convert_to_serializable(step_log)
				json.dump(step_log, f)
				f.write("\n]")


		su += np.array(rewards)
		score += sum(rewards)
		
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
					Pi[i].update(ep_states[i], ep_actions[i], all_ep_advantages)
					#try get_dist. If it fails, print the states, actions and advantages
					p = gPi[i].get_dist(np.array([obs[0][i]]))
					if np.isnan(p).any():
						print("Failed to get dist for agent", i)
						print("States", obs[0][i])
						print("Actions", ep_actions[i])
						print("Advantages", all_ep_advantages)
				else:
					gPi[i].update(ep_states[i], ep_actions[i], all_ep_advantages_saved[i])
					#try get_dist. If it fails, print the states, actions and advantages
					p = gPi[i].get_dist(np.array([obs[0][i]]))
					if np.isnan(p).any():
						print("Failed to get dist for agent", i)
						print("States", obs[0][i])
						print("Actions", ep_actions[i])
						print("Advantages", all_ep_advantages_saved[i])

			ep_actions = [[] for _ in range(n_agent)]
			ep_rewards = [[] for _ in range(n_agent)]
			ep_f_rewards = [[] for _ in range(n_agent)]
			ep_states  = [[] for _ in range(n_agent)]

			greedy=np.zeros(n_agent).astype(bool)
			for i in range(n_agent):
				greedyc = np.random.rand() <= beta
				greedy[i] = greedyc
		
		if args.render:
			M.render()
			time.sleep(0.1)
	
	# Post episode housekeeping
	print("Rewards", su)
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
			actions = []
			action_probs = []
			for i in range(n_agent):
				h = copy.deepcopy(obs[0][i])
				
				#never use greedy in validation
				more_obs = gPi[i].get_dist(np.array([h]))[0]
				h.extend(more_obs)
				if use_neighbors:
					more_obs = get_more_obs_com(True, neighbors, average_jpi, i, more_obs_size)
					h.extend(more_obs)
				# Advantages signal
				more_obs = average_jpi
				more_obs = (more_obs - np.mean(more_obs)) / (np.std(more_obs) + 0.0000000001)
				h.extend(more_obs)
				p = Pi[i].get_dist(np.array([h]))[0]
		
				if np.sum(p) != 1 or np.isnan(p).any():
					p = np.array(p)
					p = p / np.sum(p)

				actions.append(np.random.choice(M_val.n_actions, p=p))
				action_probs.append(p)
			# Take the step
			su_prev = copy.deepcopy(M_val.get_fairness_rewards())
			actions = M_val.compute_allocation(action_probs, M_val)
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
		print("Score", score)
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
	results_file = f"Results/SOTO-ILP/{args.env_name+args.env_name_mod}results.csv"
	# Also save a copy of the results file in the save_path
	results_file2 = f"{args.save_path}/results.csv"
	create=False
	if not os.path.exists("Results/SOTO-ILP"):
		os.makedirs("Results/SOTO-ILP")
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