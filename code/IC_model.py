import itertools
import random

import networkx as nx
import numpy as np

def IC(Networkx_Graph, Seed_Set, Probability):

    tree = nx.DiGraph()
    tree.add_node(Seed_Set[0])
    new_active, Ans = Seed_Set.tolist(), Seed_Set.tolist()
    while new_active:
        # Getting neighbour nodes of newly activate node
        (targets, edges) = Neighbour_finder(Networkx_Graph, Probability, new_active)
        # Calculating if any nodes of those neighbours can be activated, if yes add them to new_ones.

        new_active = []

        for (node, target) in edges:
            if np.random.uniform(0, 1) < Probability[node, target]:
                if target not in Ans: #success infected
                    tree.add_edge(node, target)
                    new_active.append(target)
                    Ans.append(target)
        # Checking which ones in new_ones are not in our Ans...only adding them to our Ans so that no duplicate in Ans.

    return Ans, tree


def Neighbour_finder(g, p, new_active):
    targets = []
    edges = []
    for node in new_active:
        node_neighbors = list(g.neighbors(node))
        targets += node_neighbors
        for target in node_neighbors:
            edges.append((node,target))

    return (targets, edges)



