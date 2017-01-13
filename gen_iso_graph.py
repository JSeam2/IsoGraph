"""
Generate adjacency matrices of isomorphic graphs

task
1) Check the sizes of isomorphic graphs you want to generate
2) Store them in different numpy files for various graph sizes

Build a table for example:
______________________________________
| Graph 1 | Graph 2 | Is Isomorphic?  |
|--------------------------------------
|   ...   |  ...    |  0 - No; 1 - Yes|
|______________________________________
    .          .            .
    .          .            .
    .          .            .

"""

import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism


def gen_rnd_graph(n):
    """
    Generate a random pair of isomorphic graphs as adjacency matrices
    Adjacency matrices are numpy arrays

    n gives the total number of nodes in the graph

    If graphs are isomorphic:
        put 1 in the Is Isomorphic column

    else:
        put 0 in the Is Isomorphic column

    output the | Graph_1 | Graph_2 |
    Output the isomorphic graphs adjacency matrix

    Some mathematical definition:
    G ≅ H (G is isomorphic to H)

    iff ∃ a: V(G)→ V(H)    (A bijection)
    such that
    a(u)a(v) ∈ E(H) ↔ uv ∈ E(G)

    Similarly,
    for some permutation matrix P,
    G ≅ H  ↔ A_G = P* A_H *P_transpose

    param:
        nodes(int): number of node

    output:
        numpy matrix
    """

    # Generate random graph, G1
    G1 = nx.dense_gnm_random_graph(n, n)

    # Generate random graph, G2
    G2 = nx.dense_gnm_random_graph(n, n)

    # Check if graphs are isomorphic
    GM = isomorphism.GraphMatcher(G1, G2)

    # Check if graphs are isomorphic
    if GM.is_isomorphic() == True:
        is_GM_isomorphic = 1

    else:
        is_GM_isomorphic = 0

    # Convert graphs to numpy matrix
    G1_numpy = nx.to_numpy_matrix(G1)
    G2_numpy = nx.to_numpy_matrix(G2)

    # Combine the G1_numpy and G2_numpy and is_GM_isomorphic


    # after creating the graph need find a way to find an isomorphic graph to
    # this randomly generating graphs won't allow you to find isomorphic pairs

    # temporary return for checking
    return is_GM_isomorphic


def save_graph(graph):
    """
    save the graphs in to a npz file
    label it respectively and place it into a ./isomorphic graphs folder
    """
    pass

if __name__ == "__main__":
    count = 0
    for x in range (100000):
        a = gen_rnd_graph(10)
        if a == 1:
            count += 1

    print("{} isomorphic graphs found".format(count))
        #print(gen_rnd_graph(10))

    # ignore anything below 3 nodes
    # num_nodes = [x for x in range(3, 21)]
