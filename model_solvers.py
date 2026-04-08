"""Build and solve the Steiner tree formulations used in the benchmark study.

This module contains the four formulations benchmarked by the repository and a
small amount of shared metric-collection logic.

The cut formulations (`UC_solve` and `DC_solve`) rely on Gurobi callbacks and
NetworkX minimum-cut computations to separate violated connectivity constraints.
The flow formulations (`UF_solve` and `DF_solve`) encode connectivity directly
with flow variables, so no lazy cuts are needed there.
"""

import time

import networkx as nx
from gurobipy import *


ROOT_NODE = 0

STATUS_NAMES = {
    GRB.LOADED: "loaded",
    GRB.OPTIMAL: "optimal",
    GRB.INFEASIBLE: "infeasible",
    GRB.INF_OR_UNBD: "inf_or_unbd",
    GRB.UNBOUNDED: "unbounded",
    GRB.CUTOFF: "cutoff",
    GRB.ITERATION_LIMIT: "iteration_limit",
    GRB.NODE_LIMIT: "node_limit",
    GRB.TIME_LIMIT: "time_limit",
    GRB.SOLUTION_LIMIT: "solution_limit",
    GRB.INTERRUPTED: "interrupted",
    GRB.NUMERIC: "numeric",
    GRB.SUBOPTIMAL: "suboptimal",
    GRB.USER_OBJ_LIMIT: "user_obj_limit",
}


def _new_stats():
    """Create the statistics dictionary attached to each solved model."""
    return {
        "root_lp_bound": None,
        "root_lp_callback_calls": 0,
        "mipnode_calls": 0,
        "mipsol_calls": 0,
        "separation_calls": 0,
        "separation_time_sec": 0.0,
        "lazy_cuts_added": 0,
    }


def _record_root_lp_bound(model, stats):
    """Record the LP bound at the root node when it is available."""
    if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
        return
    if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
        stats["root_lp_bound"] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
        stats["root_lp_callback_calls"] += 1


def _attach_stats(model, stats):
    """Attach custom benchmark stats to the solved Gurobi model and return it.

    The rest of the pipeline expects solver functions to return the `Model`
    object, because that object already carries the standard Gurobi outputs
    such as status, objective value, bounds, node count, and runtime. Attaching
    `stats` to the model lets the code keep that simple return type while still
    preserving callback-side measurements like separation counts and root-LP data.
    """
    model._benchmark_stats = stats
    return model


def _collect_stats(model):
    """Read attached statistics and fill any missing keys with defaults."""
    stats = dict(getattr(model, "_benchmark_stats", {}))
    for key, default in _new_stats().items():
        stats.setdefault(key, default)
    return stats


def DC_solve(
    nodes,
    edges,
    terminals,
    edgeCosts,
    timeLimit=30,
    threads=1,
    cuts=None,
    seed=0,
    output_flag=0,
):
    """Solve the directed cut formulation with lazy separation.

    Variables:
    - `x[u, v]` decides whether the undirected edge is selected.
    - `y[u, v]` decides whether the directed arc is available in the support graph.

    Callback logic:
    - Build a directed capacitated graph from the current `y` values.
    - For each non-root terminal, compute a minimum cut from the root.
    - If the cut capacity is below 1, add a violated lazy cut.
    """
    root = ROOT_NODE
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()

    DCmodel = Model("Directed Cut")
    DCmodel.params.OutputFlag = output_flag
    DCmodel.params.lazyConstraints = 1
    DCmodel.modelSense = GRB.MINIMIZE
    DCmodel.params.Cuts = 0 if cuts is None else cuts
    DCmodel.params.timeLimit = timeLimit
    if threads is not None:
        DCmodel.params.Threads = threads
    if seed is not None:
        DCmodel.params.Seed = seed

    x = {}
    for u, v in edges:
        x[u, v] = DCmodel.addVar(
            name=f"x#{u}#{v}",
            vtype=GRB.BINARY,
            obj=edgeCosts[u, v],
        )

    # We now create the associated digraph variables.
    # The directed support graph duplicates each undirected edge into two arcs,
    # and linking constraints ensure the orientation variables cannot be active
    # unless the underlying undirected edge is selected.
    y = {}
    for u, v in arcs:
        y[u, v] = DCmodel.addVar(name=f"y{u, v}", vtype=GRB.BINARY)
    for u, v in edges:
        DCmodel.addConstr(y[u, v] + y[v, u] <= x[u, v])

    def callback(model, where):
        # This time we split x and y variables since the minimum root-k-cut
        # value is computed according to the y-variables (arcs).
        if where == GRB.callback.MIPSOL:
            # Case 1: called for every integer solution, so we check feasibility.
            stats["mipsol_calls"] += 1
            solution_x = {
                edge: value
                for edge, value in zip(x.keys(), DCmodel.cbGetSolution(list(x.values())))
            }
            solution_y = {
                arc: value
                for arc, value in zip(y.keys(), DCmodel.cbGetSolution(list(y.values())))
            }
        elif where == GRB.callback.MIPNODE and DCmodel.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            # Case 2: called for fractional solutions, so violated cuts can be added.
            stats["mipnode_calls"] += 1
            _record_root_lp_bound(DCmodel, stats)
            solution_x = {
                edge: value
                for edge, value in zip(x.keys(), DCmodel.cbGetNodeRel(list(x.values())))
            }
            solution_y = {
                arc: value
                for arc, value in zip(y.keys(), DCmodel.cbGetNodeRel(list(y.values())))
            }
        else:
            if where == GRB.callback.MIPNODE:
                stats["mipnode_calls"] += 1
                _record_root_lp_bound(DCmodel, stats)
            # Otherwise, state that we do not currently have a usable solution.
            solution_x = None
            solution_y = None

        if solution_x is not None:
            # Case 1 or 2: rebuild the capacitated flow graph from the current
            # solution so NetworkX can evaluate root-terminal cuts.
            digraph = nx.DiGraph()
            digraph.add_nodes_from(nodes)
            digraph.add_edges_from(arcs)

            # We set arc capacities according to the current y-values.
            for u, v in edges:
                digraph.edges[u, v]["capacity"] = solution_y[u, v]
                digraph.edges[v, u]["capacity"] = solution_y[v, u]

            # We now compute the minimum root-k-cut value for each non-root terminal.
            for k in terminals:
                if k != root:
                    stats["separation_calls"] += 1
                    sep_start = time.perf_counter()
                    value, (rootPart, kPart) = nx.algorithms.flow.minimum_cut(
                        digraph, root, k
                    )
                    stats["separation_time_sec"] += time.perf_counter() - sep_start
                    if value < 1 - 1e-5:
                        # If the cut value is clearly below 1, add a violated
                        # Steiner cut using only arcs that leave the root side.
                        DCmodel.cbLazy(
                            quicksum(
                                y[u, v]
                                for u, v in arcs
                                if (u in rootPart and v in kPart)
                            )
                            >= 1
                        )
                        stats["lazy_cuts_added"] += 1

    DCmodel.optimize(callback)
    return _attach_stats(DCmodel, stats)


def DF_solve(
    nodes,
    edges,
    terminals,
    edgeCosts,
    timeLimit=30,
    threads=1,
    cuts=None,
    seed=0,
    output_flag=0,
):
    """Solve the directed multi-commodity flow formulation.

    For each non-root terminal, one unit of flow must travel from the root to
    that terminal through the directed support graph.
    """
    root = ROOT_NODE
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()

    DFmodel = Model("Directed Flow")
    DFmodel.params.OutputFlag = output_flag
    DFmodel.modelSense = GRB.MINIMIZE
    DFmodel.params.timeLimit = timeLimit
    if threads is not None:
        DFmodel.params.Threads = threads
    if cuts is not None:
        DFmodel.params.Cuts = cuts
    if seed is not None:
        DFmodel.params.Seed = seed

    x = {}
    for u, v in edges:
        x[u, v] = DFmodel.addVar(
            name="x#" + str(u) + "#" + str(v),
            vtype=GRB.BINARY,
            obj=edgeCosts[u, v],
        )

    f = {}
    for k in terminals:
        if k != root:
            for (u, v) in arcs:
                f[k, u, v] = DFmodel.addVar(
                    name="f#" + str(k) + "#" + str(u) + "#" + str(v),
                    lb=0.0,
                    ub=1.0,
                )

    # We now create the associated digraph variables.
    # `y` indicates which arc orientation is available for terminal flows.
    y = {}
    for u, v in arcs:
        y[u, v] = DFmodel.addVar(name=f"y{u, v}", vtype=GRB.BINARY)
    for u, v in edges:
        DFmodel.addConstr(y[u, v] + y[v, u] <= x[u, v])

    DFmodel.update()

    for k in terminals:
        if k != root:
            # One conservation system is created for each non-root terminal.
            for v in nodes:
                if v == root:
                    rhs = -1  # only outgoing flow is selected on the root side
                elif v == k:
                    rhs = 1  # only incoming flow is selected for terminal k
                else:
                    rhs = 0  # flow conservation holds at every other node
                DFmodel.addConstr(
                    quicksum(f[k, s, t] for (s, t) in arcs if t == v)
                    - quicksum(f[k, s, t] for (s, t) in arcs if s == v)
                    == rhs
                )
            for u, v in edges:
                # In this formulation the flow variables are associated with
                # directed arcs rather than undirected edges.
                DFmodel.addConstr(f[k, u, v] <= y[u, v])
                DFmodel.addConstr(f[k, v, u] <= y[v, u])

    def callback(model, where):
        if where == GRB.callback.MIPSOL:
            stats["mipsol_calls"] += 1
        elif where == GRB.callback.MIPNODE:
            stats["mipnode_calls"] += 1
            _record_root_lp_bound(model, stats)

    DFmodel.optimize(callback)
    return _attach_stats(DFmodel, stats)


def UC_solve(
    nodes,
    edges,
    terminals,
    edgeCosts,
    timeLimit=30,
    threads=1,
    cuts=None,
    seed=0,
    output_flag=0,
):
    """Solve the undirected cut formulation with lazy separation.

    The model selects undirected edges. The callback interprets the current edge
    values as capacities and checks whether every non-root terminal is separated
    from the root by a cut of capacity at least 1.
    """
    root = ROOT_NODE
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()

    UCmodel = Model("Undirected Cut")
    UCmodel.params.OutputFlag = output_flag
    UCmodel.params.lazyConstraints = 1
    UCmodel.modelSense = GRB.MINIMIZE
    UCmodel.params.timeLimit = timeLimit
    if threads is not None:
        UCmodel.params.Threads = threads
    if cuts is not None:
        UCmodel.params.Cuts = cuts
    if seed is not None:
        UCmodel.params.Seed = seed

    x = {}
    for u, v in edges:
        x[u, v] = UCmodel.addVar(
            name="x#" + str(u) + "#" + str(v),
            vtype=GRB.BINARY,
            obj=edgeCosts[u, v],
        )
        UCmodel.update()

    def callback(UCmodel, where):
        if where == GRB.callback.MIPSOL:
            # Case 1: called for every integer solution, so we check feasibility.
            stats["mipsol_calls"] += 1
            solution = {
                edge: value
                for edge, value in zip(
                    list(x.keys()),
                    UCmodel.cbGetSolution(list(x.values())),
                )
            }
        elif where == GRB.callback.MIPNODE and UCmodel.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
            # Case 2: called for fractional solutions, so violated cuts can be added.
            stats["mipnode_calls"] += 1
            _record_root_lp_bound(UCmodel, stats)
            solution = {
                edge: value
                for edge, value in zip(
                    list(x.keys()),
                    UCmodel.cbGetNodeRel(list(x.values())),
                )
            }
        else:
            if where == GRB.callback.MIPNODE:
                stats["mipnode_calls"] += 1
                _record_root_lp_bound(UCmodel, stats)
            # Otherwise, state that we do not currently have a usable solution.
            solution = None

        if solution is not None:
            # Rebuild the capacitated graph from the current edge values.
            digraph = nx.DiGraph()
            digraph.add_nodes_from(nodes)
            digraph.add_edges_from(arcs)
            for u, v in edges:
                digraph.edges[u, v]["capacity"] = solution[u, v]
                digraph.edges[v, u]["capacity"] = solution[u, v]
            # We now compute the minimum root-k-cut value for each non-root terminal.
            for k in terminals:
                if k != root:
                    stats["separation_calls"] += 1
                    sep_start = time.perf_counter()
                    value, (kPart, rootPart) = nx.algorithms.flow.minimum_cut(
                        digraph, root, k
                    )
                    stats["separation_time_sec"] += time.perf_counter() - sep_start
                    if value < 0.99:
                        # If the cut value is below 1, add a violated Steiner
                        # cut. In the undirected model either crossing direction
                        # corresponds to selecting the same undirected edge.
                        UCmodel.cbLazy(
                            quicksum(
                                x[u, v]
                                for u, v in edges
                                if (u in rootPart and v in kPart)
                                or (u in kPart and v in rootPart)
                            )
                            >= 1
                        )
                        stats["lazy_cuts_added"] += 1

    UCmodel.optimize(callback)
    return _attach_stats(UCmodel, stats)


def UF_solve(
    nodes,
    edges,
    terminals,
    edgeCosts,
    timeLimit=30,
    threads=1,
    cuts=None,
    seed=0,
    output_flag=0,
):
    """Solve the undirected multi-commodity flow formulation.

    For each non-root terminal, one unit of flow must travel from the root to
    that terminal. An undirected edge can carry flow in both directions, but
    only if its binary selection variable is active.
    """
    root = ROOT_NODE
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()

    UFmodel = Model("Undirected Flow")
    UFmodel.params.OutputFlag = output_flag
    UFmodel.modelSense = GRB.MINIMIZE
    UFmodel.params.timeLimit = timeLimit
    if threads is not None:
        UFmodel.params.Threads = threads
    if cuts is not None:
        UFmodel.params.Cuts = cuts
    if seed is not None:
        UFmodel.params.Seed = seed

    x = {}
    for u, v in edges:
        x[u, v] = UFmodel.addVar(
            name="x#" + str(u) + "#" + str(v),
            vtype=GRB.BINARY,
            obj=edgeCosts[u, v],
        )

    f = {}
    for k in terminals:
        if k != root:
            for (u, v) in arcs:
                f[k, u, v] = UFmodel.addVar(
                    name="f#" + str(k) + "#" + str(u) + "#" + str(v),
                    lb=0.0,
                    ub=1.0,
                )

    UFmodel.update()

    for k in terminals:
        if k != root:
            for v in nodes:
                if v == root:
                    rhs = -1  # only outgoing flow is selected on the root side
                elif v == k:
                    rhs = 1  # only incoming flow is selected for terminal k
                else:
                    rhs = 0  # flow conservation holds at every other node
                UFmodel.addConstr(
                    quicksum(f[k, s, t] for (s, t) in arcs if t == v)
                    - quicksum(f[k, s, t] for (s, t) in arcs if s == v)
                    == rhs
                )
            for u, v in edges:
                UFmodel.addConstr(f[k, u, v] <= x[u, v])
                UFmodel.addConstr(f[k, v, u] <= x[u, v])

    def callback(model, where):
        if where == GRB.callback.MIPSOL:
            stats["mipsol_calls"] += 1
        elif where == GRB.callback.MIPNODE:
            stats["mipnode_calls"] += 1
            _record_root_lp_bound(model, stats)

    UFmodel.optimize(callback)
    return _attach_stats(UFmodel, stats)


FORMULATION_REGISTRY = {
    "uc": {
        "name": "undirected cut",
        "solver": UC_solve,
        "uses_separation": True,
    },
    "uf": {
        "name": "undirected flow",
        "solver": UF_solve,
        "uses_separation": False,
    },
    "dc": {
        "name": "directed cut",
        "solver": DC_solve,
        "uses_separation": True,
    },
    "df": {
        "name": "directed flow",
        "solver": DF_solve,
        "uses_separation": False,
    },
}


def get_formulation_registry():
    """Return the formulation metadata used by the benchmark runner."""
    return FORMULATION_REGISTRY


def solve_formulation(
    formulation_key,
    nodes,
    edges,
    terminals,
    edge_costs,
    *,
    time_limit=30,
    threads=1,
    cuts=None,
    seed=0,
    output_flag=0,
):
    """Dispatch to the solver associated with one formulation key."""
    if formulation_key not in FORMULATION_REGISTRY:
        raise ValueError(f"Unknown formulation key: {formulation_key}")
    solver = FORMULATION_REGISTRY[formulation_key]["solver"]
    return solver(
        nodes,
        edges,
        terminals,
        edge_costs,
        timeLimit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
    )


def collect_model_metrics(model):
    """Extract a uniform metric dictionary from a solved Gurobi model."""
    stats = _collect_stats(model)
    status_code = model.Status
    metrics = {
        "status_code": status_code,
        "status_name": STATUS_NAMES.get(status_code, str(status_code)),
        "sol_count": model.SolCount,
        "objective": model.ObjVal if model.SolCount > 0 else None,
        "best_bound": model.ObjBound,
        "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        "bb_nodes": model.NodeCount,
        "model_runtime_sec": model.Runtime,
        "simplex_iterations": model.IterCount,
        "barrier_iterations": model.BarIterCount,
        "num_vars": model.NumVars,
        "num_constrs": model.NumConstrs,
    }
    metrics.update(stats)

    incumbent = metrics["objective"]
    root_lp_bound = metrics.get("root_lp_bound")
    # Reporting a relative gap keeps this metric comparable across instances with
    # different objective scales.
    if incumbent not in (None, 0) and root_lp_bound is not None:
        metrics["root_lp_gap"] = (incumbent - root_lp_bound) / abs(incumbent)
    else:
        metrics["root_lp_gap"] = None

    return metrics
