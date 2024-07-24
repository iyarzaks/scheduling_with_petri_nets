# pyomo.environ provides the framework for build the model
from itertools import product

import pyomo.environ as pyo

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
    # opt_glpk = pyo.SolverFactory("scip", executable="/usr/local/opt/scip/bin/scip")
    opt_glpk = pyo.SolverFactory("gurobi")
    # opt_glpk = pyo.SolverFactory(
    #     "cplex", executable="/Users/iyarzaks/Downloads/ampl.macos64/cplex"
    # )
    # opt_glpk = pyo.SolverFactory("cbc")
    # opt_glpk = pyo.SolverFactory(
    #     "scip",
    # )
    # optimizer = pyo.SolverFactory["cbc"]
    # opt_glpk.options["threads"] = 8
    # opt_glpk.options["tmlim"] = 50000
    # opt_glpk.options["mipgap"] = 0
    results = opt_glpk.solve(model)
    # results_display = model.display()
    # print(results_display)
    # print(results)
    # ask the solver to solve the model
    # results.write()
    return {"solved": True, "makespan": results["Problem"][0]["Lower bound"]}


def main():
    solve_rcpsp_optimizer(
        "/Users/iyarzaks/PycharmProjects/scheduling_with_petri_nets/extract_problems/data/j30.sm.tgz/j3039_5.sm"
    )


def solve_rcpsp_optimizer(path):
    p, u, e, c = extract_rcpsp_for_solver(path)
    res = solve_rcpsp(p, u, e, c)
    return res


if __name__ == "__main__":
    main()
