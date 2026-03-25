#from steinerDataRandom import *
import matplotlib.pyplot as plt
import networkx as nx
from gurobipy import *
from steinerDataRandom import *


root = 0
arcs = edges + [ (v,u) for (u,v) in edges ]

UFmodel = Model("Undirected Flow")
UFmodel.modelSense = GRB.MINIMIZE

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
      UFmodel.addConstr( quicksum( f[k,s,t] for (s,t) in arcs if t == v )
        - quicksum( f[k,s,t] for (s,t) in arcs if s == v ) == rhs )
    for u,v in edges:
      UFmodel.addConstr( f[k,u,v] <= x[u,v] )
      UFmodel.addConstr( f[k,v,u] <= x[u,v] )

UFmodel.optimize()

####################################################
#   SOLUTION REPRESENTATION
####################################################

show=True
if UFmodel.Status == GRB.OPTIMAL and show==True:
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
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(width/8, height/8))
    nx.draw(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths,node_size=400, node_color="skyblue", node_shape="o")
    nx.draw(G, pos, nodelist=terminals, node_size=800, node_shape="s", node_color="green")
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Steiner Tree: Selected edges highlighted in red")
    plt.axis("off")
    plt.show()
elif  not UFmodel.Status == GRB.OPTIMAL :
   print("Problem is Infeasible")
else:
   None
