"""
Processes the graph data to feed in to a neural network

Will attempt to address 9 nodes first

I don't think this is fast need some advice on this...
"""

import sqlite3

def get_data(path ='./graph.db', node = 9):
    """
    1) connects to DB location
    2) Get data from specific node number
    3) extracts data for use in neural network
    """
    conn = sqlite3.connect(path)
    c = conn.cursor()

    result = c.execute("SELECT * FROM Node_{}".format(node))
    return result.fetchall()


if __name__ == "__main__":
    a = get_data()
    #print(a[0][1])
