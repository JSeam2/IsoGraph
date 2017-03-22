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
import sqlite3

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

    :param:
        nodes(int): number of node

    :returns:
        tuple (graph1(numpy), graph2(numpy), is_isomorphic(int))
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

    return (G1_numpy, G2_numpy, is_GM_isomorphic)


def save_graph(nodes, num_graph, db_path = "./graph.db" ):
    """
    Looks for graph.db, creates graph.db if it doesn't exist
    Run gen_rnd_graph(nodes), creates up till the nodes in parameters. Doesn't create for 3 nodes and below.
    and store it with sqlite3


    :param: nodes := number of nodes the database will make until(int)
    :param: num_graph := number of graphs to generate (int)
    :param: db_path := path of sqlite3 db, default is same directory as gen_iso_graph.py
    """

    # in event connection to database is not possible put None
    conn = None

    # connect to db path
    # will make sql database if it doesn't exist
    conn = sqlite3.connect(db_path)

    with conn:
        # 1st loop to make various tables with various nodes x
        # 2nd loop to make insert gen_rnd_graph entries with nodes x
        for x in range(3,nodes):
            cur = conn.cursor()
            # Create Table this string formatting of a SQL command is generally
            # bad but we can make do with this for now.
            cur.execute("CREATE TABLE IF NOT EXISTS Node_{} (Id INT, Graph1 BLOB, Graph2 BLOB, is_isomorphic INT)".format(str(x)))

            for num in range(num_graph):
                g1, g2 , is_isomorphic = gen_rnd_graph(x)

                # Convert np tostring
                # To retrieve back using np.fromstring(bytearray)
                cur.execute("INSERT INTO Node_{} VALUES(?,?,?,?)".format(str(x))
                            ,(num, g1.tostring(), g2.tostring(), is_isomorphic))

        conn.commit()

if __name__ == "__main__":
    save_graph(10, 20000, "./graph.db")
