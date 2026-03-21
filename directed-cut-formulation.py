import networkx as nx # For graphs
import matplotlib.pyplot as plt # For drawing
import random # For random number generation
import math # For square roots
from gurobipy import *
###################################################################
# CREATING OUR STEINER TREE PROBLEM
###################################################################
numNodes = 18 # Number of nodes of the graph
numTerminals = 10 # Among them: number of terminals
width,height = 100,40 # Range of the node coordinates
edgeDistance = 20 # Nodes with at most this distance will be connected.

nodes = list(range(numNodes))
terminals = list(range(numTerminals))
G = nx.Graph()
for v in nodes:
  isTerminal = v < numTerminals
  G.add_node(v, pos=(random.randint(1, width), random.randint(1, height)), terminal=isTerminal)

edges = []
edgeCosts = {}
while nx.is_connected(G) == False:
  for u in nodes:
    for v in nodes:
      if u < v:
        dx = G.nodes[u]['pos'][0] - G.nodes[v]['pos'][0]
        dy = G.nodes[u]['pos'][1] - G.nodes[v]['pos'][1]
        distance = math.sqrt(dx*dx + dy*dy)
        if distance <= edgeDistance:
          edges.append((u,v))
          edgeCosts[(u,v)] = random.randint(8,12) * distance
          G.add_edge(u,v)
  edgeDistance += 1

print('Random graph has %s nodes and %s edges.' % (numNodes, len(edges)))
plt.figure(figsize=(width/5, height/5))
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, edgelist=edges, node_size=400, node_shape='o', width=1, node_color='blue')
nx.draw(G, pos, nodelist=terminals, node_size=800, edgelist=None, node_shape='s', node_color='red') 
plt.show()
###################################################################
# MODELLING THE STEINER TREE PROBLEM USING THE DIRECTED CUT FORMULATION
###################################################################
root = 0
arcs = edges + [ (v,u) for (u,v) in edges ]

model = Model("Directed Cut")
model.params.lazyConstraints = 1
model.modelSense = GRB.MINIMIZE
model.params.cuts = 3
model.params.timeLimit = 120

# We first create the edge variables.
x = {}
for u,v in edges:
  x[u,v] = model.addVar(name='x#'+str(u)+'#'+str(v), vtype=GRB.BINARY, obj=edgeCosts[u,v])

# We now create the associated digraph variables
y={}
for u,v in arcs:
  y[u,v]=model.addVar(name=f"y{u,v}",vtype=GRB.BINARY)
for u,v in edges:
  model.addConstr(y[u,v]+y[v,u]<= x[u,v])
#############################################################################à
# SEPARATIION PROBLEM
#############################################################################à
def callback(model, where):

  # This is required to access the corresponding variables from outside the function.
  global x
  global nodes
  global terminals
  global edges
  global root
  global arcs

  # We test for different purposes of being called.

  # Case 1: Called for every integer solution - we need to check whether it is feasible.
  if where == GRB.callback.MIPSOL:
    solution = { edge: value for edge,value in zip(x.keys(), model.cbGetSolution(list(x.values()))) }

  # Case 2: Called for fractional solutions - we can add violated inequalities.
  elif where == GRB.callback.MIPNODE and model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.OPTIMAL:
    solution = { edge: value for edge,value in zip(x.keys(), model.cbGetNodeRel(list(x.values()))) }

  # Otherwise, state that we don't have a solution.
  else:
    solution = None

  # Case 1 or 2: solution is a dictionary that maps edges to their x-entries.
  if solution is not None:
    # We create the flow graph.
    digraph = nx.DiGraph()
    digraph.add_nodes_from(nodes)
    digraph.add_edges_from(arcs)

    # We set the arc capacities according to the solution vector.
    for u,v in edges:
      digraph.edges[u,v]['capacity'] = solution[u,v]
      digraph.edges[v,u]['capacity'] = solution[u,v]

    # We now compute the minimum root-k-cut value for each non-root terminal k.
    for k in terminals:
      if k != root:
        # We call a minimum s-t-cut algorithm from the networkx graph library.
        (value,(kPart,rootPart)) = nx.algorithms.flow.minimum_cut(digraph, k, root)

        # If the cut value is (clearly) less than 1, we add a violated Steiner cut constaint.
        if value < 0.99:
          model.cbLazy(quicksum( y[u,v] for u,v in arcs if (u in rootPart and v in kPart) or (u in kPart and v in rootPart) ) >= 1)

model.optimize(callback)
selected_edges = [ (u,v) for (u,v) in edges if x[u,v].x > 0.5 ]
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
nx.draw(G, pos, edgelist=G.edges(), edge_color=edge_colors, width=edge_widths,
        node_size=400, node_color="skyblue", node_shape="o")
nx.draw(G, pos, nodelist=terminals, node_size=800, node_shape="s", node_color="green")
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("Steiner Tree: Selected edges highlighted in red")
plt.axis("off")
plt.show()