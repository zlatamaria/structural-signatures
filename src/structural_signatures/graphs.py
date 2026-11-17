"""Hard shortcut membership and maximum weak-component cycle rank."""

import networkx as nx


def shortcuts(rows, cfg):
    """A saved directed row already points in the dominant direction."""
    return rows.loc[
        (rows.cooccurrence_support >= cfg.transition_min_support)
        & (rows.ordering_consistency >= cfg.consistency)
        & (rows.adjacent_forward_count > 0)
        & (rows.nonadjacent_forward_count > 0)
    ]


def excess(graph):
    """Cycle rank for one connected, simple undirected graph."""
    return max(graph.number_of_edges() - graph.number_of_nodes() + 1, 0) if graph else 0


def most_cross_linked(edges):
    """Return the highest-excess WCC, using E then V for visualization ties."""
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    if not graph:
        return graph, 0
    undirected = graph.to_undirected(as_view=True)

    def rank(nodes):
        component = undirected.subgraph(nodes)
        return excess(component), component.number_of_edges(), len(nodes)

    nodes = max(nx.connected_components(undirected), key=rank)
    return graph.subgraph(nodes).copy(), rank(nodes)[0]


def maximum_excess(edges):
    """Only topology matters for the selector. Endpoint pairs are deduplicated."""
    graph = nx.Graph()
    graph.add_edges_from(edges)
    maximum = 0
    for nodes in nx.connected_components(graph):
        edge_count = sum(graph.degree(node) for node in nodes) // 2
        maximum = max(maximum, edge_count - len(nodes) + 1)
    return maximum
