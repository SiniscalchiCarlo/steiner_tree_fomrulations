from steinerDataRandom import *
#from steinerDataRead import *
import networkx as nx
from gurobipy import *
import  matplotlib.pyplot as plt
root = 0
arcs = edges + [ (v,u) for (u,v) in edges ]
show=True

UCmodel = Model("Undirected Cut")
UCmodel.params.lazyConstraints = 1
UCmodel.modelSense = GRB.MINIMIZE

x = {}
for u,v in edges:
  x[u,v] = UCmodel.addVar(name='x#'+str(u)+'#'+str(v), vtype=GRB.BINARY, obj=edgeCosts[u,v])
UCmodel.update()

def callback(UCmodel, where):
  global x
  global nodes
  global terminals
  global edges
  global root
  global arcs

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
        (value,(rootPart,kPart)) = nx.algorithms.flow.minimum_cut(digraph, root, k)
        #print(value, kPart, rootPart)
        if value < 0.99:
          print(quicksum( x[u,v] for u,v in edges if (u in rootPart and v in kPart) or (u in kPart and v in rootPart) ) >= 1)
          UCmodel.cbLazy(quicksum( x[u,v] for u,v in edges if (u in rootPart and v in kPart) or (u in kPart and v in rootPart) ) >= 1)

UCmodel.optimize(callback)

####################################################
#   SOLUTION REPRESENTATION
####################################################

if UCmodel.Status == GRB.OPTIMAL and show==True:
    selected_edges = [ (u,v) for (u,v) in edges if x[u,v].X > 0.5 ]
    selected_set = set(selected_edges)

    edge_colors = []
    edge_widths = []
    for u, v in G.edges():
        if (u, v) in selected_set or (v, u) in selected_set:
            edge_colors.append("red")
            edge_widths.append(3)
        else:
            edge_colors.append("lightgray")
            edge_widths.append(1)

    plt.figure(figsize=(width/5, height/5))
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths,node_size=400, node_color="skyblue", node_shape="o")
    nx.draw(G, pos, nodelist=terminals, node_size=800, node_shape="s", node_color="green")
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Steiner Tree: Selected edges highlighted in red")
    plt.axis("off")
    plt.show()
elif  not UCmodel.Status == GRB.OPTIMAL :
   print("Problem is Infeasible")
else:
   None