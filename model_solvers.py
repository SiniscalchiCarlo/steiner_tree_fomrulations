import time
from collections import deque
from gurobipy import GRB, Model, quicksum


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


def _apply_common_params(
    model,
    *,
    time_limit,
    threads,
    cuts,
    seed,
    output_flag,
    lazy_constraints=False,
):
    model.Params.OutputFlag = output_flag
    model.ModelSense = GRB.MINIMIZE
    model.Params.TimeLimit = time_limit
    if threads is not None:
        model.Params.Threads = threads
    if cuts is not None:
        model.Params.Cuts = cuts
    if seed is not None:
        model.Params.Seed = seed
    if lazy_constraints:
        model.Params.LazyConstraints = 1


def _record_root_lp_bound(model, stats):
    if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
        return

    node_count = model.cbGet(GRB.Callback.MIPNODE_NODCNT)
    if node_count == 0:
        stats["root_lp_bound"] = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
        stats["root_lp_callback_calls"] += 1


def _make_passive_callback(stats):
    def callback(model, where):
        if where == GRB.Callback.MIPNODE:
            stats["mipnode_calls"] += 1
            _record_root_lp_bound(model, stats)
        elif where == GRB.Callback.MIPSOL:
            stats["mipsol_calls"] += 1

    return callback


def _minimum_cut(nodes, arcs, capacities, source, sink, tolerance=1e-9):
    adjacency = {node: [] for node in nodes}
    residual = {}

    for u, v in arcs:
        if v not in adjacency[u]:
            adjacency[u].append(v)
        if u not in adjacency[v]:
            adjacency[v].append(u)
        residual[u, v] = residual.get((u, v), 0.0) + capacities.get((u, v), 0.0)
        residual.setdefault((v, u), 0.0)

    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in parent:
                    continue
                if residual.get((current, neighbor), 0.0) <= tolerance:
                    continue
                parent[neighbor] = current
                queue.append(neighbor)

        if sink not in parent:
            break

        path_flow = float("inf")
        node = sink
        while parent[node] is not None:
            prev = parent[node]
            path_flow = min(path_flow, residual[prev, node])
            node = prev

        node = sink
        while parent[node] is not None:
            prev = parent[node]
            residual[prev, node] -= path_flow
            residual[node, prev] = residual.get((node, prev), 0.0) + path_flow
            node = prev

    reachable = set()
    queue = deque([source])
    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        for neighbor in adjacency[current]:
            if neighbor not in reachable and residual.get((current, neighbor), 0.0) > tolerance:
                queue.append(neighbor)

    cut_value = 0.0
    for u in reachable:
        for v in adjacency[u]:
            if v not in reachable:
                cut_value += capacities.get((u, v), 0.0)

    return cut_value, reachable, set(nodes) - reachable


def _separate_min_cuts(model, edge_vars, solution, nodes, edges, arcs, terminals, stats):
    stats["separation_calls"] += 1
    start = time.perf_counter()

    capacities = {}
    for u, v in edges:
        capacity = solution[u, v]
        capacities[u, v] = capacity
        capacities[v, u] = capacity

    for terminal in terminals:
        if terminal == ROOT_NODE:
            continue
        cut_value, root_part, terminal_part = _minimum_cut(
            nodes,
            arcs,
            capacities,
            ROOT_NODE,
            terminal,
        )
        if cut_value < 0.99:
            model.cbLazy(
                quicksum(
                    edge_vars[u, v]
                    for u, v in edges
                    if (u in root_part and v in terminal_part)
                    or (u in terminal_part and v in root_part)
                )
                >= 1
            )
            stats["lazy_cuts_added"] += 1

    stats["separation_time_sec"] += time.perf_counter() - start


def _make_cut_callback(stats, edge_vars, nodes, edges, arcs, terminals):
    keys = list(edge_vars.keys())
    values = list(edge_vars.values())

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            stats["mipsol_calls"] += 1
            solution_values = model.cbGetSolution(values)
            solution = dict(zip(keys, solution_values))
            _separate_min_cuts(model, edge_vars, solution, nodes, edges, arcs, terminals, stats)
        elif where == GRB.Callback.MIPNODE:
            stats["mipnode_calls"] += 1
            _record_root_lp_bound(model, stats)
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) != GRB.OPTIMAL:
                return
            solution_values = model.cbGetNodeRel(values)
            solution = dict(zip(keys, solution_values))
            _separate_min_cuts(model, edge_vars, solution, nodes, edges, arcs, terminals, stats)

    return callback


def _build_undirected_cut_model(
    nodes,
    edges,
    terminals,
    edge_costs,
    *,
    time_limit,
    threads,
    cuts,
    seed,
    output_flag,
):
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()
    model = Model("Undirected Cut")
    _apply_common_params(
        model,
        time_limit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
        lazy_constraints=True,
    )

    edge_vars = {}
    for u, v in edges:
        edge_vars[u, v] = model.addVar(
            name=f"y#{u}#{v}",
            vtype=GRB.BINARY,
            obj=edge_costs[u, v],
        )

    model.update()
    callback = _make_cut_callback(stats, edge_vars, nodes, edges, arcs, terminals)
    return model, callback, stats


def _build_directed_cut_model(
    nodes,
    edges,
    terminals,
    edge_costs,
    *,
    time_limit,
    threads,
    cuts,
    seed,
    output_flag,
):
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()
    model = Model("Directed Cut")
    _apply_common_params(
        model,
        time_limit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
        lazy_constraints=True,
    )

    edge_vars = {}
    for u, v in edges:
        edge_vars[u, v] = model.addVar(
            name=f"x#{u}#{v}",
            vtype=GRB.BINARY,
            obj=edge_costs[u, v],
        )

    arc_vars = {}
    for u, v in arcs:
        arc_vars[u, v] = model.addVar(name=f"y#{u}#{v}", vtype=GRB.BINARY)

    for u, v in edges:
        model.addConstr(arc_vars[u, v] + arc_vars[v, u] <= edge_vars[u, v])

    model.update()
    callback = _make_cut_callback(stats, edge_vars, nodes, edges, arcs, terminals)
    return model, callback, stats


def _build_undirected_flow_model(
    nodes,
    edges,
    terminals,
    edge_costs,
    *,
    time_limit,
    threads,
    cuts,
    seed,
    output_flag,
):
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()
    model = Model("Undirected Flow")
    _apply_common_params(
        model,
        time_limit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
    )

    edge_vars = {}
    for u, v in edges:
        edge_vars[u, v] = model.addVar(
            name=f"x#{u}#{v}",
            vtype=GRB.BINARY,
            obj=edge_costs[u, v],
        )

    flow_vars = {}
    for terminal in terminals:
        if terminal == ROOT_NODE:
            continue
        for u, v in arcs:
            flow_vars[terminal, u, v] = model.addVar(
                name=f"f#{terminal}#{u}#{v}",
                lb=0.0,
                ub=1.0,
            )

    model.update()

    for terminal in terminals:
        if terminal == ROOT_NODE:
            continue
        for node in nodes:
            if node == ROOT_NODE:
                rhs = -1
            elif node == terminal:
                rhs = 1
            else:
                rhs = 0
            model.addConstr(
                quicksum(flow_vars[terminal, u, v] for u, v in arcs if v == node)
                - quicksum(flow_vars[terminal, u, v] for u, v in arcs if u == node)
                == rhs
            )
        for u, v in edges:
            model.addConstr(flow_vars[terminal, u, v] <= edge_vars[u, v])
            model.addConstr(flow_vars[terminal, v, u] <= edge_vars[u, v])

    callback = _make_passive_callback(stats)
    return model, callback, stats


def _build_directed_flow_model(
    nodes,
    edges,
    terminals,
    edge_costs,
    *,
    time_limit,
    threads,
    cuts,
    seed,
    output_flag,
):
    arcs = edges + [(v, u) for (u, v) in edges]
    stats = _new_stats()
    model = Model("Directed Flow")
    _apply_common_params(
        model,
        time_limit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
    )

    edge_vars = {}
    for u, v in edges:
        edge_vars[u, v] = model.addVar(
            name=f"x#{u}#{v}",
            vtype=GRB.BINARY,
            obj=edge_costs[u, v],
        )

    arc_vars = {}
    for u, v in arcs:
        arc_vars[u, v] = model.addVar(name=f"y#{u}#{v}", vtype=GRB.BINARY)

    flow_vars = {}
    for terminal in terminals:
        if terminal == ROOT_NODE:
            continue
        for u, v in arcs:
            flow_vars[terminal, u, v] = model.addVar(
                name=f"f#{terminal}#{u}#{v}",
                lb=0.0,
            )

    model.update()

    for u, v in edges:
        model.addConstr(arc_vars[u, v] + arc_vars[v, u] <= edge_vars[u, v])

    for terminal in terminals:
        if terminal == ROOT_NODE:
            continue
        for node in nodes:
            rhs = 1 if node == ROOT_NODE else -1 if node == terminal else 0
            model.addConstr(
                quicksum(flow_vars[terminal, u, v] for u, v in arcs if u == node)
                - quicksum(flow_vars[terminal, u, v] for u, v in arcs if v == node)
                == rhs
            )
        for u, v in arcs:
            model.addConstr(flow_vars[terminal, u, v] <= arc_vars[u, v])

    callback = _make_passive_callback(stats)
    return model, callback, stats


FORMULATION_REGISTRY = {
    "uc": {
        "name": "undirected cut",
        "builder": _build_undirected_cut_model,
        "uses_separation": True,
    },
    "uf": {
        "name": "undirected flow",
        "builder": _build_undirected_flow_model,
        "uses_separation": False,
    },
    "dc": {
        "name": "directed cut",
        "builder": _build_directed_cut_model,
        "uses_separation": True,
    },
    "df": {
        "name": "directed flow",
        "builder": _build_directed_flow_model,
        "uses_separation": False,
    },
}


def get_formulation_registry():
    return FORMULATION_REGISTRY


def build_formulation(
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
    metadata = FORMULATION_REGISTRY[formulation_key]
    model, callback, stats = metadata["builder"](
        nodes,
        edges,
        terminals,
        edge_costs,
        time_limit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
    )
    model._benchmark_stats = stats
    model._benchmark_formulation_key = formulation_key
    model._benchmark_formulation_name = metadata["name"]
    return model, callback, stats


def collect_model_metrics(model):
    stats = dict(getattr(model, "_benchmark_stats", {}))
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
    model, callback, _ = build_formulation(
        formulation_key,
        nodes,
        edges,
        terminals,
        edge_costs,
        time_limit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
    )
    model.optimize(callback)
    return model


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
    model, _, _ = build_formulation(
        formulation_key,
        nodes,
        edges,
        terminals,
        edge_costs,
        time_limit=time_limit,
        threads=threads,
        cuts=cuts,
        seed=seed,
        output_flag=output_flag,
    )
    relaxed_model = model.relax()
    relaxed_model.Params.OutputFlag = output_flag
    relaxed_model.Params.TimeLimit = time_limit
    if threads is not None:
        relaxed_model.Params.Threads = threads
    if cuts is not None:
        relaxed_model.Params.Cuts = cuts
    if seed is not None:
        relaxed_model.Params.Seed = seed
    relaxed_model.optimize()
    return relaxed_model


def UC_solve(nodes, edges, terminals, edgeCosts, timeLimit=30, threads=1, cuts=None, seed=0):
    return solve_formulation(
        "uc",
        nodes,
        edges,
        terminals,
        edgeCosts,
        time_limit=timeLimit,
        threads=threads,
        cuts=cuts,
        seed=seed,
    )


def UF_solve(nodes, edges, terminals, edgeCosts, timeLimit=30, threads=1, cuts=None, seed=0):
    return solve_formulation(
        "uf",
        nodes,
        edges,
        terminals,
        edgeCosts,
        time_limit=timeLimit,
        threads=threads,
        cuts=cuts,
        seed=seed,
    )


def DC_solve(nodes, edges, terminals, edgeCosts, timeLimit=30, threads=1, cuts=None, seed=0):
    return solve_formulation(
        "dc",
        nodes,
        edges,
        terminals,
        edgeCosts,
        time_limit=timeLimit,
        threads=threads,
        cuts=cuts,
        seed=seed,
    )


def DF_solve(nodes, edges, terminals, edgeCosts, timeLimit=30, threads=1, cuts=None, seed=0):
    return solve_formulation(
        "df",
        nodes,
        edges,
        terminals,
        edgeCosts,
        time_limit=timeLimit,
        threads=threads,
        cuts=cuts,
        seed=seed,
    )
