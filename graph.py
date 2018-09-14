import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import csv
import sys


def graph(graph_points=None, title=None):
    if graph_points is None:
        graph_points = [('A', 'B'), ('B', 'C')]
    edges = set(graph_points)

    plt.clf()

    node_size = 800
    node_color = '#80bfff'
    node_alpha = 1.0
    node_text_size = 12
    edge_color = '#80bfff'
    edge_alpha = 1.0
    edge_thickness = 2
    edge_text_pos = 0.3
    text_font = 'sans-serif'

    nodes = set()
    for node_1, node_2 in edges:
            nodes.add(node_1)
            nodes.add(node_2)

    edges = set(graph_points)

    graph = nx.Graph()

    labels = {}
    for node in nodes:
        graph.add_node(node)
        labels[node] = "$%s$" % node

    edge_labels = {}
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
        edge_labels[(edge[0], edge[1])] = ''

    #position = nx.shell_layout(graph)
    position = graphviz_layout(graph, prog="neato")
    #nx.draw(graph, position)

    # draw graph in pieces
    nx.draw_networkx_edges(graph, position, width=edge_thickness,
                           alpha=edge_alpha, edge_color=edge_color)
    nx.draw_networkx_nodes(graph, position, node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_labels(graph, position, labels=labels, font_size=node_text_size,
                            font_family=text_font)
    nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels, alpha=0.5)

    if False:
        labels = [e.__repr__() for e in edges]

        edge_labels = dict(zip(graph, labels))
        nx.draw_networkx_edge_labels(graph, position, edge_labels=edge_labels,
                                     label_pos=edge_text_pos)

    # show graph
    if title is None:
        title = 'Variable messaging network'
    plt.title(title)
    plt.show()
    return graph


def graph_from_file(file_name):
    data = list(csv.reader(open(file_name, 'r')))
    data = [tuple(i) for i in data]
    graph(data)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        graph_from_file('messages.txt')
    else:
        graph_from_file(sys.argv[1])
