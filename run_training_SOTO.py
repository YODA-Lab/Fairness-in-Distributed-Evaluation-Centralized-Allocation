import os
import json

# add argument processing
import argparse

# masked = True
# ALF = True

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="BiasedDM")
parser.add_argument("--env_name_mod", type=str, default="")
parser.add_argument("--split", type=str, default="False")
parser.add_argument("--learn_fairness", type=str, default="True")
parser.add_argument("--learn_utility", type=str, default="True")
parser.add_argument("--multi_head", type=str, default="False")
parser.add_argument("--logging", type=str, default="True")
parser.add_argument("--render", type=str, default="False")
parser.add_argument("--fairness_type", type=str, default="variance")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--u_model_loc", type=str, default="")
parser.add_argument("--learning_beta", type=float, default=0.0)
parser.add_argument("--past_discount", type=float, default=1.0)  # Assume no past discounting
parser.add_argument("--warm_start", type=float, default=0.0) # Assume no warm start

#add some arguments for this script, but remove them before passing to train.py
parser.add_argument("--masked", type=str, default="True")
parser.add_argument("--ALF", type=str, default="True")
args = parser.parse_args()

params = vars(args)
#Handle boolean arguments
for k,v in params.items():
    if v == "True" or v == "False":
        params[k] = v == "True"

masked = params["masked"]
ALF = params["ALF"]
# remove the extra arguments
del params["masked"]
del params["ALF"]
fairness_type = params["fairness_type"]

with open(f"hyperparams_SOTO.json") as f:
    hyperparams = json.load(f)[params["env_name"]]
    for key, value in hyperparams.items():
        if key not in params:
            params[key] = value
        else:
            if params[key] == None or params[key] == "":
                params[key] = value
            else:
                print("Key ", key, " already exists in args. Ignoring value from hyperparams.json")

func = f"""python train_SOTO.py """
if ALF:
    func = f"""python train_SOTO_ALF.py """
if masked:
    func = f"""python train_SOTO_mask.py """
    if ALF:
        func = f"""python train_SOTO_mask_ALF.py """
for key, value in params.items():
    if value is None or value == "":
        continue
    if key=='learning_beta':
        print(value, key)
    func += f""" --{key} {value} """
# print(func)
os.system(func)