import gurobipy as gp
from gurobipy import GRB
import numpy as np

def get_assignment_old(Qvalues):
	n_agents = len(Qvalues)
	n_resources = len(Qvalues[0])
	#Create a model
	m = gp.Model("mip1")
	m.setParam('OutputFlag', 0)
	#Create variables
	x = m.addVars(n_agents, n_resources, vtype=GRB.BINARY, name="x")
	#Set objective
	m.setObjective(sum(sum(Qvalues[i][j]*x[i,j] for j in range(n_resources)) for i in range(n_agents)), GRB.MAXIMIZE)
	#Add constraints
	m.addConstrs(sum(x[i,j] for j in range(n_resources))==1 for i in range(n_agents)) # Each agent can only be assigned to exactly one resource
	m.addConstrs(sum(x[i,j] for i in range(n_agents))<=1 for j in range(1,n_resources)) # Each resource except the first one can only be assigned to one agent
	#Solve
	m.optimize()
	#Get solution
	assignment = []
	for i in range(n_agents):
		for j in range(n_resources):
			if x[i,j].x==1:
				assignment.append(j)
	return assignment

def get_assignment_gurobi(Qvalues, resource_counts, agent_constraints=None):
	# Qvalues is a list of lists, where each list is the Q values for each agent for each resource
	# resource_counts is a list of the number of agents that can be assigned to each resource
	# agent_constraints is a list of the resources an agent cannot be assigned to
	n_agents = len(Qvalues)
	n_resources = len(Qvalues[0])
	#Create a model
	m = gp.Model("mip1")
	m.setParam('OutputFlag', 0)
	#Create variables
	x = m.addVars(n_agents, n_resources, vtype=GRB.BINARY, name="x")
	#Set objective and add constraints
	m.setObjective(sum(sum(Qvalues[i][j]*x[i,j] for j in range(n_resources)) for i in range(n_agents)), GRB.MAXIMIZE)
	m.addConstrs(sum(x[i,j] for j in range(n_resources))==1 for i in range(n_agents)) # Each agent can only be assigned to exactly one resource (action)
	m.addConstrs(sum(x[i,j] for i in range(n_agents))<=resource_counts[j] for j in range(n_resources)) # Each resource can only be assigned to a certain number of agents
	if agent_constraints is not None:
		for i in range(n_agents):
			for j in agent_constraints[i]:
				m.addConstr(x[i,j]==0)
	#Solve
	m.optimize()
	#Get solution
	assignment = []
	for i in range(n_agents):
		for j in range(n_resources):
			if x[i,j].x==1:
				assignment.append(j)
	return assignment


from scipy.optimize import linprog

def get_assignment(Qvalues, resource_counts, agent_constraints=None):
	"""
	Speed comparison
	Gurobi: 20 seconds for 40 warmstart iterations
	SciPy: 23 seconds for 40 warmstart iterations
	PuLP: 3 minutes 38 seconds for 40 warmstart iterations
	"""
	# SciPy implementation. Decent speed, but not as fast as Gurobi
	n_agents = len(Qvalues)
	n_resources = len(Qvalues[0])

	# Objective coefficients
	c = -np.array(Qvalues).flatten()

	# Inequality constraints: each resource can only be assigned to a certain number of agents
	A_ub = np.zeros((n_resources, n_agents * n_resources))
	for j in range(n_resources):
		for i in range(n_agents):
			A_ub[j, i * n_resources + j] = 1
	b_ub = resource_counts

	# Equality constraints: each agent can only be assigned to exactly one resource
	A_eq = np.zeros((n_agents, n_agents * n_resources))
	for i in range(n_agents):
		for j in range(n_resources):
			A_eq[i, i * n_resources + j] = 1
	b_eq = np.ones(n_agents)

	# Bounds: each variable is binary (0 or 1)
	bounds = [(0, 1)] * (n_agents * n_resources)

	# If there are agent constraints, add them
	if agent_constraints:
		for i, constraints in enumerate(agent_constraints):
			for j in constraints:
				bounds[i * n_resources + j] = (0, 0)

	# Solve the linear programming problem
	result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

	if result.x is None:
		print("ENCOUNTERED INFEASIBLE PROBLEM, rounding off QValues")
		# Try with rounded off QValues to catch numerical instability
		Qvalues = Qvals = np.round(np.multiply(Qvalues, 100), 6)
		return get_assignment(Qvalues, resource_counts, agent_constraints)

	# Extract assignment from solution
	assignment = [np.argmax(result.x[i * n_resources: (i + 1) * n_resources]) for i in range(n_agents)]

	return assignment