import os
import time
import importlib.util
from gurobipy import GRB
from model_solvers import UC_solve, UF_solve, DC_solve
import pandas as pd
from tqdm import tqdm
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

for file in tqdm(os.listdir(INSTANCE_DIR)):

    if file.endswith(".py"):
        path = os.path.join(INSTANCE_DIR, file)
        nodes, edges, terminals, edgeCosts = load_instance(path)

        for model_name, solver in models.items():
            print(f" \n Solving {file} with {model_name}...")

            t0 = time.time()
            model = solver(nodes, edges, terminals, edgeCosts)
            elapsed = time.time() - t0

            results.append({
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
})
pd.DataFrame(results).to_csv("results.csv", index=False)

