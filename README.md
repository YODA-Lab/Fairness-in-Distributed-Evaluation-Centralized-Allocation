## DECAF
Code for the paper "Learning Fairness in Multi-agent Systems with Distributed-Evaluation, Centralized-Allocation", published at FAccT 2026.

## Installation
Create an environment using the environment.yml file:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate decaf
```

## Usage
To run the code, use the following command:
```bash
python run_training_DECAF.py
```

Add the following arguments to the command:
- `--env_name`: Name of the environment (default: "BiasedDM").
- `--split`: Whether to use split training for the utility and fairness models (default: "False"). If False, JO model is trained.
- `--learn_fairness`: Whether to learn the fairness model (default: "True"). If True (and split is True), SO model is trained.
- `--learn_utility`: Whether to learn the utility model (default: "True"). If False (and split is True), FO model is trained.
- `--learning_beta`: The weight of the fairness term. Defaults to 0.0 (no fairness).
- `--fairness_type`: Type of fairness metric to use (default: "variance"). Options are "variance", "alpha-fair", "maximin", and "ggf".

To train an FO model, first a joint model should be trained with beta=0, and the u_model_loc argument should be set to the location of the trained model. 
