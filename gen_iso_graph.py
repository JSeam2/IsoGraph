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

# ignore anything below 3 nodes
num_nodes = [x for x in range(3, 21)]

# generate isomorphic graphs
def gen_isograph(nodes):
    """
    Get no. of nodes as input
    Generate pairs of isomorphic graphs
    Verify the isomorphic graphs
    Output the isomorphic graphs adjacency matrix
    
    param:
        nodes(int): number of node
    
    output:
        numpy matrix
    """
    
    return graph

def save_graph(graph):
    """
    save the graphs in to a npz file
    label it respectively and place it into a ./isomorphic graphs folder


