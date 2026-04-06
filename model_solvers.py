import time
from gurobipy import *
import networkx as nx


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
    if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
        return
    if model.cbGet(GRB.Callback.MIPNODE_NODCNT) == 0:
        stats["root_lp_bound"] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
        stats["root_lp_callback_calls"] += 1


def _attach_stats(model, stats):
    model._benchmark_stats = stats
    return model


def _collect_stats(model):
    stats = dict(getattr(model, "_benchmark_stats", {}))
    for key, default in _new_stats().items():
        stats.setdefault(key, default)
    return stats

####################################################
# DIRECTED CUT
####################################################
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
    root = 0
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
        x[u, v] = DCmodel.addVar(name=f'x#{u}#{v}', vtype=GRB.BINARY, obj=edgeCosts[u, v])
    # We now create the associated digraph variables
    y={}
    for u,v in arcs:
      y[u,v]=DCmodel.addVar(name=f"y{u,v}",vtype=GRB.BINARY)
    for u,v in edges:
      DCmodel.addConstr(y[u,v]+y[v,u]<= x[u,v])
    
    def callback(model, where):
            #This time we split x and y variables since we need to calculate the minimum root-k-cut value according to the y-variables
            # Case 1: Called for every integer solution - we need to check whether it is feasible.
            if where == GRB.callback.MIPSOL:
              stats["mipsol_calls"] += 1
              solution_x = { edge: value for edge,value in zip(x.keys(), DCmodel.cbGetSolution(list(x.values()))) }
              solution_y ={ arc: value for arc,value in zip(y.keys(), DCmodel.cbGetSolution(list(y.values()))) }
            # Case 2: Called for fractional solutions - we can add violated inequalities.
            elif where == GRB.callback.MIPNODE and DCmodel.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
              stats["mipnode_calls"] += 1
              _record_root_lp_bound(DCmodel, stats)
              solution_x = { edge: value for edge,value in zip(x.keys(), DCmodel.cbGetNodeRel(list(x.values()))) }
              solution_y ={ arc: value for arc,value in zip(y.keys(), DCmodel.cbGetNodeRel(list(y.values()))) }
            # Otherwise, state that we don't have a solution.
            else:
              if where == GRB.callback.MIPNODE:
                  stats["mipnode_calls"] += 1
                  _record_root_lp_bound(DCmodel, stats)
              solution_x = None
              solution_y = None
            
            # Case 1 or 2: solution is a dictionary that maps edges to their x-entries.
            if solution_x is not None:
              # We create the flow graph.
              digraph = nx.DiGraph()
              digraph.add_nodes_from(nodes)
              digraph.add_edges_from(arcs)
            

              #  We set the arc capacities according to the solution vector associated to the y-variables this time
              for u,v in edges:
                digraph.edges[u,v]['capacity'] = solution_y[u,v]
                digraph.edges[v,u]['capacity'] = solution_y[v,u]

              # We now compute the minimum root-k-cut value for each non-root terminal k.
              for k in terminals:
                #print(k, root)
                if k != root:
                    stats["separation_calls"] += 1
                    sep_start = time.perf_counter()
                    (value,(rootPart,kPart)) = nx.algorithms.flow.minimum_cut(digraph, root, k)
                    stats["separation_time_sec"] += time.perf_counter() - sep_start
                    #print(value, kPart, rootPart)
                    if value < 1-1e-5:
                        # If the cut value is (clearly) less than 1, we add a violated Steiner cut constaint. 
                        # We keep outgoing arcs from the root side of the cut to the terminal side.
                        DCmodel.cbLazy(quicksum( y[u,v] for u,v in arcs if (u in rootPart and v in kPart)) >= 1)
                        stats["lazy_cuts_added"] += 1

        

    DCmodel.optimize(callback)
    return _attach_stats(DCmodel, stats)

##############################################################
# UNDIRECTED CUT 
##############################################################
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

    root = 0
    arcs = edges + [ (v,u) for (u,v) in edges ]
    show=False
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
    for u,v in edges:
        x[u,v] = UCmodel.addVar(name='x#'+str(u)+'#'+str(v), vtype=GRB.BINARY, obj=edgeCosts[u,v])
        UCmodel.update()

    def callback(UCmodel, where):

            if where == GRB.callback.MIPSOL:
                stats["mipsol_calls"] += 1
                solution = { edge: value for edge,value in zip(list(x.keys()), UCmodel.cbGetSolution(list(x.values()))) }
                #print(solution)
            elif where == GRB.callback.MIPNODE and UCmodel.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                stats["mipnode_calls"] += 1
                _record_root_lp_bound(UCmodel, stats)
                solution = { edge: value for edge,value in zip(list(x.keys()), UCmodel.cbGetNodeRel(list(x.values()))) }
                #print(solution)
            else:
                if where == GRB.callback.MIPNODE:
                    stats["mipnode_calls"] += 1
                    _record_root_lp_bound(UCmodel, stats)
                solution = None

            if solution is not None:
                digraph = nx.DiGraph()
                digraph.add_nodes_from(nodes)
                digraph.add_edges_from(arcs)
                for u,v in edges:
                    digraph.edges[u,v]['capacity'] = solution[u,v]
                    digraph.edges[v,u]['capacity'] = solution[u,v]
                for k in terminals:
                    #print(k, root)
                    if k != root:
                        stats["separation_calls"] += 1
                        sep_start = time.perf_counter()
                        (value,(kPart,rootPart)) = nx.algorithms.flow.minimum_cut(digraph, root, k)
                        stats["separation_time_sec"] += time.perf_counter() - sep_start
                        #print(value)
                        #print(value, kPart, rootPart)
                        if value < 0.99:
                            #print(quicksum( y[u,v] for u,v in edges if (u in rootPart and v in kPart) or (u in kPart and v in rootPart) ) >= 1)
                            UCmodel.cbLazy(quicksum( x[u,v] for u,v in edges if (u in rootPart and v in kPart) or (u in kPart and v in rootPart) ) >= 1)
                            stats["lazy_cuts_added"] += 1
    UCmodel.optimize(callback)
    return _attach_stats(UCmodel, stats)

##############################################################
# UNDIRECTED FLOW
##############################################################
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

    root = 0
    arcs = edges + [ (v,u) for (u,v) in edges ]
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
    for u,v in edges:
        x[u,v] = UFmodel.addVar(name='x#'+str(u)+'#'+str(v), vtype=GRB.BINARY, obj=edgeCosts[u,v])

    f = {}
    for k in terminals:
        if k != root:
            for (u,v) in arcs:
                f[k,u,v] = UFmodel.addVar(name='f#'+str(k)+'#'+str(u)+'#'+str(v), lb=0.0, ub=1.0)

    UFmodel.update()

    for k in terminals:
        if k != root:
            for v in nodes:
                if v == root:
                     rhs = -1
                elif v == k:
                    rhs = 1
                else:
                    rhs = 0
                UFmodel.addConstr( quicksum( f[k,s,t] for (s,t) in arcs if t == v ) - quicksum( f[k,s,t] for (s,t) in arcs if s == v ) == rhs )
            for u,v in edges:
                UFmodel.addConstr( f[k,u,v] <= x[u,v] )
                UFmodel.addConstr( f[k,v,u] <= x[u,v] )

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
}


def get_formulation_registry():
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


def solve_lp_relaxation(
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
    raise NotImplementedError(
        "Static LP relaxation is not implemented for the current solver file."
    )


def collect_model_metrics(model):
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
    if incumbent not in (None, 0) and root_lp_bound is not None:
        metrics["root_lp_gap"] = (incumbent - root_lp_bound) / abs(incumbent)
    else:
        metrics["root_lp_gap"] = None

    return metrics


