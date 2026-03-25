import networkx as nx # For graphs
import matplotlib.pyplot as plt # For drawing
import random # For random number generation
import math # For square roots
show=False

numNodes = 100  # Number of nodes of the graph
numTerminals = 2 # Among them: number of terminals
width,height = 100,40 # Range of the node coordinates
edgeDistance = 40 # Nodes with at most this distance will be connected.

nodes = list(range(numNodes))
terminals = list(range(numTerminals))
G = nx.Graph()
for v in nodes:
  isTerminal = v < numTerminals
  G.add_node(v, pos=(random.randint(1, width), random.randint(1, height)), terminal=isTerminal)

edges = []
edgeCosts = {}
for u in nodes:
  for v in nodes:
   if u < v:
     dx = G.nodes[u]['pos'][0] - G.nodes[v]['pos'][0]
     dy = G.nodes[u]['pos'][1] - G.nodes[v]['pos'][1]
     distance = math.sqrt(dx*dx + dy*dy)
     if distance <= edgeDistance:
       edges.append((u,v))
       edgeCosts[(u,v)] = math.floor(random.randint(8,12) * distance)
       G.add_edge(u,v)

print('# Random graph with %s nodes and %s edges.' % (numNodes, len(edges)))
print('nodes =', nodes)
print('terminals =', terminals)
print('edges =', edges)
print('edgeCosts =', edgeCosts)
if show==True:
  plt.figure(figsize=(width/5, height/5))
  pos = nx.get_node_attributes(G, 'pos')
  nx.draw(G, pos, edgelist=edges, node_size=400, node_shape='o', width=1, node_color='blue')
  nx.draw(G, pos, nodelist=terminals, node_size=800, edgelist=None, node_shape='s', node_color='red')

  plt.show()