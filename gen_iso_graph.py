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
import scipy as sp
import networkx as nx


# generate isomorphic graphs
def gen_isograph(n):
    """
    Get no. of nodes as input
    Generate pairs of isomorphic graphs
    Verify the isomorphic graphs
    IOutput the isomorphic graphs adjacency matrix

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
    # Generate random graph, G1 and convert to numpy matrix
    G1 = nx.dense_gnm_random_graph(n, n)
    G1_numpy = to_numpy_matrix(G1)

    # Generate random graph, G2 and convert to numpy matrix
    G2 = nx.dense_gnm_random_graph(n, n)
    G2_numpy = to_numpy_matrix(G2)

    # Combine the matrices together
    Gall = 


    # after creating the graph need find a way to find an isomorphic graph to
    # this randomly generating graphs won't allow you to find isomorphic pairs

    return graph


def save_graph(graph):
    """
    save the graphs in to a npz file
    label it respectively and place it into a ./isomorphic graphs folder
    """
    pass


if __name__ == "__main__":

    
    # ignore anything below 3 nodes
    # num_nodes = [x for x in range(3, 21)]
