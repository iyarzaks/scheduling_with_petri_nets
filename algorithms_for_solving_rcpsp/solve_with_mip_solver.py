# pyomo.environ provides the framework for build the model
from itertools import product

import gurobipy as gp
import pyomo.environ as pyo
from gurobipy import GRB

from extract_problems.extract_problem import extract_rcpsp_for_solver

# SolverFactory allows to call the solver to solve

# store the solver in a variable to call it later, we need to tell google colab
# the specific path on which the solver was installed


# activities = {"Start", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "End"}
P = [0, 5, 2, 5, 6, 5, 2, 3, 2, 4, 3, 0]  # activity durations

U = [
    [0, 0, 0],  # list of resource consumptions
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 0],
]

E = [
    [0, 1],  # list of precedence constraints
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [3, 6],
    [3, 7],
    [5, 8],
    [6, 9],
    [8, 10],
    [9, 10],
    [4, 11],
    [7, 11],
    [10, 11],
]

C = [1, 1, 1]


def solve_rcpsp(p, u, e, c):
    n = len(p) - 2
    ph = sum(p)
    (R, J, T) = (range(len(c)), range(len(p)), range(ph))
    model = pyo.ConcreteModel()
    model.J = pyo.RangeSet(0, len(p) - 1)  # set for the activities
    model.T = pyo.RangeSet(0, ph)  # set for the days in the planning horizon
    model.Xs = pyo.Var(model.J, model.T, within=pyo.Binary)  # variables
    Xs = model.Xs
    Z = sum([t * Xs[(n + 1, t)] for t in model.T])
    # we want to minimize this objective function
    model.obj = pyo.Objective(expr=Z, sense=pyo.minimize)
    model.one_xs = pyo.ConstraintList()  # create a list of constraints
    for j in model.J:
        # add constraints to the list created above
        model.one_xs.add(expr=sum(Xs[(j, t)] for t in model.T) == 1)
    model.pc = pyo.ConstraintList()  # precedence constraints
    for j, s in e:
        model.pc.add(expr=sum(t * Xs[(s, t)] - t * Xs[(j, t)] for t in model.T) >= p[j])
    model.rl = pyo.ConstraintList()  # resource level constraints
    for r, t in product(R, T):
        model.rl.add(
            expr=sum(
                u[j][r] * Xs[(j, t2)]
                for j in model.J
                for t2 in range(max(0, t - p[j] + 1), t + 1)
            )
            <= c[r]
        )
    # opt = pyo.SolverFactory(
    #     "scip", executable="/home/dsi/zaksiya/SCIPOptSuite-9.1.1-Linux/bin/scip"
    # )
    # opt = pyo.SolverFactory("gurobi")
    # opt_glpk = pyo.SolverFactory(
    #     "cplex", executable="/Users/iyarzaks/Downloads/ampl.macos64/cplex"
    # )
    opt = pyo.SolverFactory("cbc")
    # opt = pyo.SolverFactory(
    #     "scip",
    # )
    # optimizer = pyo.SolverFactory["cbc"]
    # opt_glpk.options["threads"] = 8
    # opt_glpk.options["tmlim"] = 50000
    # opt_glpk.options["mipgap"] = 0
    results = opt.solve(
        model,
        tee=False,
        # options={
        #     "LogFile": "gurobi_log.txt",  # Save log to a file
        #     "OutputFlag": 1,  # Enable output (1) or disable it (0)
        # },
    )
    # results_display = model.display()
    # print(results_display)
    # print(results)
    # ask the solver to solve the model
    # results.write()
    return {"solved": True, "makespan": results["Problem"][0]["Lower bound"]}


def solve_rcpsp_lp_relaxation_optimized(p, u, e, c, finished_activities):
    # Precompute sets and parameters
    R, J = len(c), len(p)
    active_activities = [j for j in range(J) if str(j) not in finished_activities]

    # If all activities are finished, return 0
    if not active_activities:
        return 0

    # Adjust the planning horizon to only consider unfinished activities
    ph = sum(p[j] for j in active_activities)
    T = range(ph)

    # Create a new model
    model = gp.Model("RCPSP_LP_Relaxation")

    # Create variables only for active activities
    Xs = model.addVars(active_activities, T, vtype=GRB.CONTINUOUS, name="Xs")

    # Set objective (consider the last unfinished activity)
    last_active = max(active_activities)
    model.setObjective(gp.quicksum(t * Xs[last_active, t] for t in T), GRB.MINIMIZE)

    # Add constraints
    for j in active_activities:
        model.addConstr(gp.quicksum(Xs[j, t] for t in T) == 1)

    # Adjust precedence constraints
    for j, s in e:
        if j in active_activities and s in active_activities:
            model.addConstr(gp.quicksum(t * (Xs[s, t] - Xs[j, t]) for t in T) >= p[j])

    # Adjust resource constraints
    for r in range(R):
        for t in T:
            model.addConstr(
                gp.quicksum(
                    u[j][r] * Xs[j, t2]
                    for j in active_activities
                    for t2 in range(max(0, t - p[j] + 1), t + 1)
                )
                <= c[r]
            )

    # Set Gurobi parameters
    model.setParam("OutputFlag", 0)  # Suppress output
    model.setParam("Method", 1)  # Use dual simplex method
    model.setParam("Presolve", 2)  # Aggressive presolve

    # Optimize model
    model.optimize()
    return model.objVal


def solve_rcpsp_lp_relaxation(p, u, e, c, finished_activities):
    # Update the planning horizon (ph)
    ph = sum(p)
    (R, J, T) = (range(len(c)), range(len(p)), range(ph))

    # Remove completed activities
    completed_activities = [int(j) for j in finished_activities]
    active_activities = [j for j in J if j not in completed_activities]

    model = pyo.ConcreteModel()
    model.J = pyo.Set(initialize=active_activities)  # set for the remaining activities
    model.T = pyo.RangeSet(0, ph)  # set for the days in the planning horizon
    model.Xs = pyo.Var(
        model.J, model.T, within=pyo.NonNegativeReals, bounds=(0, 1)
    )  # relaxed variables
    Xs = model.Xs

    # Objective function: minimize the completion time of the project
    Z = sum([t * Xs[(len(p) - 1, t)] for t in model.T])
    model.obj = pyo.Objective(expr=Z, sense=pyo.minimize)

    # Each activity must be scheduled exactly once
    model.one_xs = pyo.ConstraintList()
    for j in model.J:
        model.one_xs.add(expr=sum(Xs[(j, t)] for t in model.T) == 1)

    # Precedence constraints
    model.pc = pyo.ConstraintList()
    for j, s in e:
        if j in active_activities and s in active_activities:
            model.pc.add(
                expr=sum(t * Xs[(s, t)] - t * Xs[(j, t)] for t in model.T) >= p[j]
            )

    # Resource level constraints
    model.rl = pyo.ConstraintList()
    for r, t in product(R, T):
        model.rl.add(
            expr=sum(
                u[j][r] * Xs[(j, t2)]
                for j in model.J
                for t2 in range(max(0, t - p[j] + 1), t + 1)
            )
            <= c[r]
        )
    opt_glpk = pyo.SolverFactory("gurobi")
    results = opt_glpk.solve(model)
    return results["Problem"][0]["Lower bound"]


def main():
    solve_rcpsp_optimizer("../extract_problems/data/j30.sm.tgz/j304_2.sm")


def solve_rcpsp_optimizer(path):
    p, u, e, c = extract_rcpsp_for_solver(path)
    res = solve_rcpsp(p, u, e, c)
    # res = solve_rcpsp_lp_relaxation(
    #     rcpsp_base=None,
    #     finished_activities=[],
    #     started_activities=None,
    #     current_time=None,
    #     job_finish_activity=None,
    #     alternatives=None,
    #     heuristic_params=(p, u, e, c),
    # )
    return res


# if __name__ == "__main__":
#     main()
