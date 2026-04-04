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
from DFRL.common.utils import eligibility_traces, default_config, make_env, RunningMeanStd, str2bool, discount_rewards
from DFRL.common.ppo_independant import PPOPolicyNetworkMasked, ValueNetwork
from collections import deque
# Use CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

normalize_inputs = True
save_path = "logs/FEN-Mask/"
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

#FEN params
LAMBDA = -1 if args.env_name == "Job" else 0.97
GAMMA = 0.98
meta_skip_etrace = True
lr_actor = train_args.learning_rate
max_us = {'BiasedDM': 1, 'JobAlloc': 1, 'Job': 1, 'Plant': 1, 'Matthew': 1}
max_u = max_us[args.env_name]/args.max_steps
n_signals = {'BiasedDM': 4, 'JobAlloc': 4, 'Job': 4, 'Plant': 4, 'Matthew': 4}
n_signal = n_signals[args.env_name]

config = {"agent":
		  {
			  "ggi_type":"xpowerminusn", 
			  "lambda":LAMBDA, 
			  "lr_actor":lr_actor,
			  "ggi_constant":2
			}
		}

n_agent = M.n_agents
n_episode = args.n_episode
args.fairness_type = 'ggf'
fairness_function = fairness_router(args.fairness_type)
# fairness_function = fairness_router(cofvar)

obs = M.get_obs()
num_features = len(obs[0][0])

T = 25 if args.env_name in ['Job', 'JobAlloc', 'BiasedDM'] else 50
meta_Pi = []
meta_V = []
Pi = [[] for _ in range(n_agent)]
V = [[] for _ in range(n_agent)]

shared_nets = True
if shared_nets:
	#sharing the network
	meta_Pi_shared = PPOPolicyNetworkMasked(num_features=num_features+2, num_actions=n_signal, layer_size=128, epsilon=0.1, learning_rate=lr_actor)
	meta_V_shared = ValueNetwork(num_features=num_features+2, hidden_size=128, learning_rate=0.001)
	Pi_shared = [[] for _ in range(n_agent)]
	V_shared = [[] for _ in range(n_agent)]
	for i in range(n_agent):
		for j in range(n_signal):
			Pi_shared[i].append(PPOPolicyNetworkMasked(num_features=num_features, num_actions=M.n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor))
			V_shared[i].append(ValueNetwork(num_features=num_features, hidden_size=256, learning_rate=0.001))
	for i in range(n_agent):
		meta_Pi.append(meta_Pi_shared)
		meta_V.append(meta_V_shared)
		for j in range(n_signal):
			Pi[i].append(Pi_shared[i][j])
			V[i].append(V_shared[i][j])
else:
	for i in range(n_agent):
		meta_Pi.append(PPOPolicyNetworkMasked(num_features=num_features+2, num_actions=n_signal, layer_size=128, epsilon=0.1, learning_rate=lr_actor))
		meta_V.append(ValueNetwork(num_features=num_features+2, hidden_size=128, learning_rate=0.001))
		for j in range(n_signal):
			Pi[i].append(PPOPolicyNetworkMasked(num_features=num_features, num_actions=M.n_actions, layer_size=256, epsilon=0.1, learning_rate=lr_actor))
			V[i].append(ValueNetwork(num_features=num_features, hidden_size=256, learning_rate=0.001))

if normalize_inputs:
	meta_obs_rms = [RunningMeanStd(shape=2) for _ in range(n_agent)]

episode_util_logs = []
run_metrics = {'system_utility':[], 'fairness':[], 'min_utility':[], 'objective':[],'variance':[]}

for i_episode in range(1, args.n_episode+1):

	avg = [0] * n_agent
	u_bar = [0] * n_agent
	utili = [0] * n_agent
	u = [[] for _ in range(n_agent)]

	ep_actions = [[] for _ in range(n_agent)]
	ep_rewards = [[] for _ in range(n_agent)]
	ep_states  = [[] for _ in range(n_agent)]
	ep_f_rewards = [[] for _ in range(n_agent)]
	ep_masks = [[] for _ in range(n_agent)]

	meta_z = [[] for _ in range(n_agent)]
	meta_rewards = [[] for _ in range(n_agent)]
	meta_states = [[] for _ in range(n_agent)]

	signal = [0] * n_agent
	rat = [0.0] * n_agent

	score = 0
	steps = 0
	su = [0.] * n_agent
	su = np.array(su)

	M.reset()
	obs = M.get_obs()

	# Run the episode
	for steps in range(1, args.max_steps+1):

		if (steps-1) % T == 0:
			for i in range(n_agent):
				h = copy.deepcopy(obs[0][i])
				h.append(rat[i])
				h.append(utili[i])
				if normalize_inputs:
					h[-2:] = list(meta_obs_rms[i].obs_filter(np.array(h)[-2:]))
				p_z = meta_Pi[i].get_dist(np.array([h]))[0]
				z = np.random.choice(n_signal, p=p_z)
				signal[i] = z
				meta_z[i].append(to_categorical(z, n_signal))
				meta_states[i].append(h)

		step_log = {"agents":{}} #log for each step
		
		# Random agent order
		actions = [None for _ in range(n_agent)]
		action_probs = [None for _ in range(n_agent)]
		agent_order = np.random.permutation(n_agent)
		for i in agent_order:
			h = copy.deepcopy(obs[0][i])
			illegal_actions = M.get_illegal_actions(i, actions)
			mask = np.ones(M.n_actions)
			mask[illegal_actions] = 0
			p = Pi[i][signal[i]].get_dist(np.array([h]), mask=np.array([mask]))[0]
			ep_states[i].append(h)

			actions[i] = np.random.choice(M.n_actions, p=p)
			action_probs[i] = p

			step_log['agents'][int(i)] = {'state':h, 'action':actions[i], 'probs':p, 'mask':mask}
			ep_masks[i].append(mask)

		# Take the step
		su_prev = copy.deepcopy(M.get_fairness_rewards())
		rewards = M.step(actions)
		obs = M.get_obs()
		if M.fairness_vars=='':
			fairness_rewards = rewards
		else:
			su_post = copy.deepcopy(M.get_fairness_rewards())
			fairness_rewards = [su_post[i] - su_prev[i] for i in range(n_agent)]

		su += np.array(rewards)
		score += sum(rewards)
		
		for i in range(n_agent):
			# u[i].append(rewards[i])
			u[i].append(fairness_rewards[i]) # If using frewards
			u_bar[i] = sum(u[i]) / len(u[i])
		
		for i in range(n_agent):
			avg[i] = sum(u_bar) / len(u_bar)
			if avg[i] != 0:
				rat[i] = (u_bar[i] - avg[i]) / avg[i]
			else:
				rat[i] = 0.0
			if max_u != None:
				utili[i] = min(1, avg[i] / max_u)
			else:
				utili[i] = avg[i]

		for i in range(n_agent):
			ep_actions[i].append(to_categorical(actions[i], M.n_actions))
			if signal[i] == 0:
				ep_rewards[i].append(rewards[i])
				ep_f_rewards[i].append(fairness_rewards[i])
			else:
				h = copy.deepcopy(obs[0][i])
				h.append(rat[i])
				h.append(utili[i])
				if normalize_inputs:
					h[-2:] = list(meta_obs_rms[i].obs_filter(np.array(h)[-2:]))
				p_z = meta_Pi[i].get_dist(np.array([h]))[0]
				r_p = p_z[signal[i]]
				ep_rewards[i].append(r_p)
				ep_f_rewards[i].append(r_p)

		# Update agent policy
		if steps % T == 0:
			for i in range(n_agent):
				meta_rewards[i].append(utili[i] / (0.1 + abs(rat[i])))
				ep_actions[i] = np.array(ep_actions[i])
				ep_rewards[i] = np.array(ep_rewards[i], dtype=np.float_)
				ep_f_rewards[i] = np.array(ep_f_rewards[i], dtype=np.float_)
				ep_states[i] = np.array(ep_states[i])
				ep_masks[i] = np.array(ep_masks[i])

				if LAMBDA < -0.1:
					targets = discount_rewards(ep_rewards[i], GAMMA)
					V[i][signal[i]].update(ep_states[i], targets)
					vs = V[i][signal[i]].get(ep_states[i])
				else:
					vs = V[i][signal[i]].get(ep_states[i])
					targets = eligibility_traces(ep_rewards[i], vs, V[i][signal[i]].get(copy.deepcopy([obs[0][i]])), GAMMA, LAMBDA)
					V[i][signal[i]].update(ep_states[i], targets)
				ep_advantages = targets - vs
				ep_advantages = (ep_advantages - np.mean(ep_advantages)) / (np.std(ep_advantages) + 0.0000000001)
				Pi[i][signal[i]].update(ep_states[i], ep_actions[i], ep_advantages, ep_masks[i])

			ep_actions = [[] for _ in range(n_agent)]
			ep_rewards = [[] for _ in range(n_agent)]
			ep_f_rewards = [[] for _ in range(n_agent)]
			ep_states  = [[] for _ in range(n_agent)]
			ep_masks = [[] for _ in range(n_agent)]
		
		if args.render:
			M.render()
			time.sleep(0.1)
	
	# Post episode housekeeping
	print("Rewards", su)
	episode_util_logs.append(['train', i_episode, M.su, M.get_fairness_rewards(), M.discounted_su])
	epi_metrics = add_epi_metrics_to_logs(summary_writer, M.su, None, 0.0, i_episode, args.max_steps, verbose=True, prefix="", logging=args.logging, fair_rewards=M.get_fairness_rewards(), fairness_type=args.fairness_type, fairness_function=fairness_function)
	
	for key, value in epi_metrics.items():
		run_metrics[key].append(value)
		#Print the average metrics
		if i_episode%50==0:
			print("Average "+key+": ", np.mean(run_metrics[key]))
	
	# Update meta networks
	for i in range(n_agent):
		if len(meta_rewards[i]) == 0:
			continue
		meta_z[i] = np.array(meta_z[i])
		meta_rewards[i] = np.array(meta_rewards[i])
		meta_states[i] = np.array(meta_states[i])
		# if done:
		# 	meta_states[i] = meta_states[i][:len(meta_rewards[i]), :]
		# 	meta_z[i] = meta_z[i][:len(meta_rewards[i]), :]
		meta_vs = meta_V[i].get(meta_states[i])
		if meta_skip_etrace:
			meta_targets = meta_rewards[i]
		else:
			h = copy.deepcopy(obs[0][i])
			h.append(rat[i])
			h.append(utili[i])
			meta_targets = eligibility_traces(meta_rewards[i], meta_vs, meta_V[i].get([h]), GAMMA, LAMBDA)
		meta_V[i].update(meta_states[i], meta_targets)
		meta_advantages = meta_targets - meta_vs
		meta_Pi[i].update(meta_states[i], meta_z[i], meta_advantages)

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
		obs = M_val.get_obs()
		score = 0
		signal = [0] * n_agent
		rat = [0.0] * n_agent
		utili = [0] * n_agent
		u = [[] for _ in range(n_agent)]
		u_bar = [0] * n_agent
		avg = [0] * n_agent
		su = [0.] * n_agent
		su = np.array(su)

		for steps in range(args.max_steps):
			# get meta action
			if (steps-1) % T == 0:
				for i in range(n_agent):
					h = copy.deepcopy(obs[0][i])
					h.append(rat[i])
					h.append(utili[i])
					if normalize_inputs:
						h[-2:] = list(meta_obs_rms[i].obs_filter(np.array(h)[-2:]))
					p_z = meta_Pi[i].get_dist(np.array([h]))[0]
					z = np.random.choice(n_signal, p=p_z)
					signal[i] = z
			
			agent_order = np.random.permutation(n_agent)
			actions = [None for _ in range(n_agent)]
			for i in agent_order:
				h = copy.deepcopy(obs[0][i])
				illegal_actions = M_val.get_illegal_actions(i, actions)
				mask = np.ones(M_val.n_actions)
				mask[illegal_actions] = 0
				p = Pi[i][signal[i]].get_dist(np.array([h]), mask=np.array([mask]))[0]
				actions[i] = np.random.choice(M_val.n_actions, p=p)

			# Take the step
			su_prev = copy.deepcopy(M_val.get_fairness_rewards())
			rewards = M_val.step(actions)
			obs = M_val.get_obs()
			score += sum(rewards)
			if M_val.fairness_vars=='':
				fairness_rewards = rewards
			else:
				su_post = copy.deepcopy(M_val.get_fairness_rewards())
				fairness_rewards = [su_post[i] - su_prev[i] for i in range(n_agent)]

			for i in range(n_agent):
				u[i].append(fairness_rewards[i])
				u_bar[i] = sum(u[i]) / len(u[i])
			for i in range(n_agent):
				avg[i] = sum(u_bar) / len(u_bar)
				if avg[i] != 0:
					rat[i] = (u_bar[i] - avg[i]) / avg[i]
				else:
					rat[i] = 0.0
				if max_u != None:
					utili[i] = min(1, avg[i] / max_u)
				else:
					utili[i] = avg[i]
			
		val_util_logs.append(['val', val_eps, M_val.su, M_val.get_fairness_rewards(), M_val.discounted_su])
		print("Score", score)
		print(M_val.su)
		print(M_val.get_fairness_rewards(), "Fair")
		print(M_val.discounted_su, "Fair")

		metrics = get_metrics_from_rewards(M_val.su, args.learning_beta, fair_rewards=M_val.get_fairness_rewards(), fairness_type=args.fairness_type, fairness_function=fairness_function)
		for key, value in metrics.items():
			print(key, value)
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
	results_file = f"Results/FEN-Mask/{args.env_name+args.env_name_mod}results.csv"
	# Also save a copy of the results file in the save_path
	results_file2 = f"{args.save_path}/results.csv"
	create=False
	if not os.path.exists("Results/FEN-Mask"):
		os.makedirs("Results/FEN-Mask")
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