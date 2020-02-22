"""
Functions related to node ordering.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
from numpy.testing import assert_equal


def get_node_to_depth(T, root):
    node_to_depth = {root : 0}
    for head, tail in nx.bfs_edges(T, root):
        node_to_depth[tail] = node_to_depth[head] + 1
    return node_to_depth


def get_node_to_subtree_depth(T, root):
    subdepth = {}
    for node in reversed(list(nx.bfs_nodes(T, root))):
        successors = T.successors(tail)
        if not successors:
            subdepth[node] = 0
        else:
            subdepth[node] = max(subdepth[s] for s in successors) + 1
    return subdepth


def get_node_to_subtree_thickness(T, root):
    thickness = {}
    for node in nx.dfs_postorder_nodes(T, root):
        successors = list(T.successors(node))
        if not successors:
            thickness[node] = 1
        else:
            w = sorted((thickness[n] for n in successors), reverse=True)
            thickness[node] = max(w + i for i, w in enumerate(w))
    return thickness


def get_node_evaluation_order(T, root):
    thickness = get_node_to_subtree_thickness(T, root)
    expanded = set()
    stack = [root]
    while stack:
        n = stack.pop()
        if n in expanded:
            yield n
        else:
            successors = list(T.successors(n))
            if not successors:
                yield n
            else:
                expanded.add(n)
                pairs = [(thickness[x], x) for x in successors]
                progeny = [x for w, x in sorted(pairs)]
                stack.extend([n] + progeny)
