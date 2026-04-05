from gurobipy import *
import networkx as nx

####################################################
# DIRECTED CUT
####################################################
def DC_solve(nodes, edges, terminals, edgeCosts, timeLimit=30):
    root = 0
    arcs = edges + [(v, u) for (u, v) in edges]

    DCmodel = Model("Directed Cut")
    DCmodel.params.OutputFlag = 0
    DCmodel.params.lazyConstraints = 1
    DCmodel.modelSense = GRB.MINIMIZE
    DCmodel.params.Cuts = 0

    DCmodel.params.timeLimit = timeLimit
    

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
              solution_x = { edge: value for edge,value in zip(x.keys(), DCmodel.cbGetSolution(list(x.values()))) }
              solution_y ={ arc: value for arc,value in zip(y.keys(), DCmodel.cbGetSolution(list(y.values()))) }
            # Case 2: Called for fractional solutions - we can add violated inequalities.
            elif where == GRB.callback.MIPNODE and DCmodel.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
              solution_x = { edge: value for edge,value in zip(x.keys(), DCmodel.cbGetNodeRel(list(x.values()))) }
              solution_y ={ arc: value for arc,value in zip(y.keys(), DCmodel.cbGetNodeRel(list(y.values()))) }
            # Otherwise, state that we don't have a solution.
            else:
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
                    (value,(kPart,rootPart)) = nx.algorithms.flow.minimum_cut(digraph, root, k)
                    #print(value, kPart, rootPart)
                    if value < 1-1e-5:
                        # If the cut value is (clearly) less than 1, we add a violated Steiner cut constaint. 
                        # We only keep the outgoins arcs that has tail in r and head in k 
                        DCmodel.cbLazy(quicksum( y[u,v] for u,v in arcs if (u in rootPart and v in kPart)) >= 1)

        

    DCmodel.optimize(callback)
    return DCmodel

##############################################################
# UNDIRECTED CUT 
##############################################################
def UC_solve(nodes, edges, terminals, edgeCosts, timeLimit=30):

    root = 0
    arcs = edges + [ (v,u) for (u,v) in edges ]
    show=False

    UCmodel = Model("Undirected Cut")
    UCmodel.params.OutputFlag = 0
    UCmodel.params.lazyConstraints = 1
    UCmodel.modelSense = GRB.MINIMIZE
    UCmodel.params.timeLimit = timeLimit

    x = {}
    for u,v in edges:
        x[u,v] = UCmodel.addVar(name='x#'+str(u)+'#'+str(v), vtype=GRB.BINARY, obj=edgeCosts[u,v])
        UCmodel.update()

    def callback(UCmodel, where):

            if where == GRB.callback.MIPSOL:
                solution = { edge: value for edge,value in zip(list(x.keys()), UCmodel.cbGetSolution(list(x.values()))) }
                #print(solution)
            elif where == GRB.callback.MIPNODE and UCmodel.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
                solution = { edge: value for edge,value in zip(list(x.keys()), UCmodel.cbGetNodeRel(list(x.values()))) }
                #print(solution)
            else:
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
                        (value,(kPart,rootPart)) = nx.algorithms.flow.minimum_cut(digraph, root, k)
                        #print(value)
                        #print(value, kPart, rootPart)
                        if value < 0.99:
                            #print(quicksum( y[u,v] for u,v in edges if (u in rootPart and v in kPart) or (u in kPart and v in rootPart) ) >= 1)
                            UCmodel.cbLazy(quicksum( x[u,v] for u,v in edges if (u in rootPart and v in kPart) or (u in kPart and v in rootPart) ) >= 1)
    UCmodel.optimize(callback)
    return UCmodel

##############################################################
# UNDIRECTED FLOW
##############################################################
def UF_solve(nodes, edges, terminals, edgeCosts, timeLimit=30):

    root = 0
    arcs = edges + [ (v,u) for (u,v) in edges ]

    UFmodel = Model("Undirected Flow")
    UFmodel.params.OutputFlag = 0
    UFmodel.modelSense = GRB.MINIMIZE
    UFmodel.params.timeLimit = timeLimit
    

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

    UFmodel.optimize()
    return UFmodel


