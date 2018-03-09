# -*- coding: utf-8 -*-
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

import os
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import pandas

def gen_rnd_graph(n, mode="dense"):
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

    :param:
        nodes(int): number of node
        mode(str) : 'dense' to generate dense graph
                    'sparse' for sparse graph

    :returns:
        tuple (graph1(numpy), graph2(numpy), is_isomorphic(int))
    """

    if mode == 'dense':
        # Generate random graph, G1
        G1 = nx.dense_gnm_random_graph(n, n)

        # Generate random graph, G2
        G2 = nx.dense_gnm_random_graph(n, n)

    # This might not really be sparse
    elif mode == 'sparse':
        G1 = nx.gnm_random_graph(n, n)
        G2 = nx.gnm_random_graph(n, n)

    elif mode == 'binomial':
        G1 = nx.gnp_random_graph(3, 0.5)
        G2 = nx.gnp_random_graph(3, 0.5)

    else:
        return 'Invalid Mode'

    # Check if graphs are isomorphic
    GM = isomorphism.GraphMatcher(G1, G2)

    # Check if graphs are isomorphic
    if GM.is_isomorphic():
        is_GM_isomorphic = 1

    else:
        is_GM_isomorphic = 0

    # Convert graphs to numpy matrix
    G1_numpy = nx.to_numpy_matrix(G1)
    G2_numpy = nx.to_numpy_matrix(G2)

    return (G1_numpy, G2_numpy, is_GM_isomorphic)



if __name__ == "__main__":
    # to read out upper diagonal
    #print(A[np.triu_indices(A.shape[0],1)])
    N = 3
    G1 = []
    G2 = []
    is_iso = []

    for i in range(10000):
        A,B,C = (gen_rnd_graph(N, mode='binomial'))
        G1.append(A)
        G2.append(B)
        is_iso.append(C)

    df = pandas.DataFrame(data = {"G1" : G1, "G2" : G2, "is_iso" : is_iso})
    df.to_pickle("{}_node_adjacency.pkl".format(N))


