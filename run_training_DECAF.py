import os
import json

# add argument processing
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env_name", type=str, default="BiasedDM")
parser.add_argument("--env_name_mod", type=str, default="")
parser.add_argument("--split", type=str, default="False")
parser.add_argument("--learn_fairness", type=str, default="True")
parser.add_argument("--learn_utility", type=str, default="True")
parser.add_argument("--logging", type=str, default="True")
parser.add_argument("--fairness_type", type=str, default="variance")
parser.add_argument("--tag", type=str, default="")
parser.add_argument("--u_model_loc", type=str, default="")
parser.add_argument("--learning_beta", type=float, default=0.0)
args = parser.parse_args()

params = vars(args)
#Handle boolean arguments
for k,v in params.items():
    if v == "True" or v == "False":
        params[k] = v == "True"

fairness_type = params["fairness_type"]

with open(f"hyperparams.json") as f:
    hyperparams = json.load(f)[params["env_name"]]
    for key, value in hyperparams.items():
        if key not in params:
            params[key] = value
        else:
            if params[key] == None or params[key] == "":
                params[key] = value
            else:
                print("Key ", key, " already exists in args. Ignoring value from hyperparams.json")


if not params["learn_utility"]:
    if params["multi_head"]:
        print("Multihead not supported for learn_utility=False")
        exit()
    if params["u_model_loc"] == "":
        print("Must provide u_model_loc when learn_utility=False")
        print("Looking for default model in the Joint directory")
        pth = "logs/"+params["env_name"]+params["env_name_mod"]+"/"+params["tag"]
        if params['simple_obs']==True:
            pth = pth+"/Simple"
        # model_loc = pth+f"/variance/Joint/0.0/1/models/best/best_model.ckpt"
        # choose a random bootstrap id
        # bootstrap_ids = os.listdir(pth+f"/variance/Joint/0.0/")
        bootstrap_ids = os.listdir(pth+f"/{fairness_type}/Joint/0.0/")
        import random
        bootstrap_id = random.choice(bootstrap_ids)
        model_loc = pth+f"/{fairness_type}/Joint/0.0/{bootstrap_id}/models/best/best_model.ckpt"
        if not os.path.exists(model_loc):
            print("Model not found in ", model_loc)
            exit()
        params["u_model_loc"] = model_loc
    print("Loading model from ", params["u_model_loc"])
else:
    if params["u_model_loc"] != "":
        print("Ignoring u_model_loc when learn_utility=True")
    params["u_model_loc"] = ""


func = f"""python train_DDQN.py """
for key, value in params.items():
    if value is None or value == "":
        continue
    if key=='learning_beta':
        print(value, key)
    func += f""" --{key} {value} """
# print(func)
os.system(func)