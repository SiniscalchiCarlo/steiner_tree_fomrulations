import os
import time
import importlib.util
from gurobipy import GRB
from model_solvers import UC_solve, UF_solve, DC_solve
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
def load_instance(path):
    spec = importlib.util.spec_from_file_location("instance", path) 
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nodes, module.edges, module.terminals, module.edgeCosts

INSTANCE_DIR = r"C:\Users\david\OneDrive - uniroma1.it\Desktop\MILP\steiner_tree_fomrulations\Istances"

models = {
    "undirected cut": UC_solve,
    "undirected flow": UF_solve,
    "directed cut": DC_solve,
}

results = []

def run_single(args):
    file, model_name, solver_fn, instance_dir = args
    path = os.path.join(instance_dir, file)
    nodes, edges, terminals, edgeCosts = load_instance(path)
    t0 = time.time()
    model = solver_fn(nodes, edges, terminals, edgeCosts)
    elapsed = time.time() - t0
    return{
    "instance": file,
    "model": model_name,
    "obj": model.ObjVal if model.SolCount > 0 else None,
    "time": round(elapsed, 2),
    "status": model.Status,

    # Punto 2 - qualità LP al root node
    "lp_relaxation": model.ObjBound,        # bound LP
    "mip_gap": model.MIPGap if model.SolCount > 0 else None,

    # Punto 3 - nodi branch and bound
    "bb_nodes": model.NodeCount,

    # Punto 4 - separation: devi misurarlo dentro la callback
    #"sep_time": ...,   # vedi sotto
    #"sep_calls": ...,  # vedi sotto

    # Punto 5 - metadati istanza
    "n_nodes": len(nodes),
    "n_edges": len(edges),
    "n_terminals": len(terminals),
}

if __name__ == "__main__":
    tasks = [
        (file, model_name, solver, INSTANCE_DIR)
        for file in os.listdir(INSTANCE_DIR) if file.endswith(".py")
        for model_name, solver in models.items()
    ]

    with Pool(processes=4) as pool:
        results = list(tqdm(pool.imap(run_single, tasks), total=len(tasks)))

    pd.DataFrame(results).to_csv("results.csv", index=False)