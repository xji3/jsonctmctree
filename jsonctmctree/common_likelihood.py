"""
Code that is common for computing likelihood-related functions.

This is shared between the code that computes log likelihood and derivatives
and the code that computes posterior expectations of things like
linear combinations of labeled transitions on edges.

"""
from __future__ import division, print_function, absolute_import

import sys

import numpy as np
from numpy.testing import assert_equal

from .node_ordering import get_node_evaluation_order

__all__ = [
        'create_indicator_array',
        'get_conditional_likelihoods',
        'get_subtree_likelihoods',
        ]


def create_indicator_array(
        node,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations):
    """
    Create the initial array indicating observations.

    Allow fine-grained missing data by using -1 as a placeholder
    for a missing observation.
    If all information about one of the variables in the multivariate process
    is missing at one of the nodes, then a coarser-grained representation
    of the missingness should be used.

    Parameters
    ----------
    node : x
        x
    state_space_shape : x
        x
    observable_nodes : x
        x
    observable_axes : x
        x
    iid_observations : x
        x

    Returns
    -------
    obs : 2d ndarray of shape (nstates, nsites)
        Observation indicator array.

    """
    nstates = np.prod(state_space_shape)
    nsites, nobservables = iid_observations.shape
    state_space_ndim = len(state_space_shape)
    state_space_axes = range(state_space_ndim)

    # Initialize the active array, initially in a high dimensional shape.
    # This array is large; for data with many iid sites,
    # such active arrays dominate the memory usage of the program.
    obs_shape = (nsites, ) + tuple(state_space_shape)
    obs = np.ones(obs_shape, dtype=float)

    # For each observable associated with the node under consideration,
    # apply the observation mask across all iid sites.
    local_observables = np.flatnonzero(observable_nodes == node)
    for idx in local_observables:
        states = iid_observations[:, idx]
        axis = observable_axes[idx]
        k = state_space_shape[axis]
        projection_shape = [k if i == axis else 1 for i in state_space_axes]
        mask_shape = (nsites, ) + tuple(projection_shape)
        indicator_arrays = np.zeros((k+1, k), dtype=int)
        np.fill_diagonal(indicator_arrays, 1)
        indicator_arrays[-1, :] = 1
        obs *= np.take(indicator_arrays, states, axis=0).reshape(mask_shape)

    # Reshape the observation array to 2d.
    # First collapse the dimensionality of the state space to 1d,
    # then transpose the array so that it has shape (nstates, nsites),
    # to prepare for P.dot(obs) where P has shape (nstates, nstates).
    obs = obs.reshape((nsites, np.prod(state_space_shape))).T

    # Return the observation indicator array.
    assert_equal(obs.shape, (nstates, nsites))
    return obs


def get_subtree_likelihoods(
        f,
        store_all,
        T, root, edges, edge_rate_pairs, edge_process_pairs,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations,
        ):
    """
    Compute likelihood arrays associated with nodes.

    Unlike get_conditional_likelihoods, this function
    does not look at the upstream edge of the node.

    The shape of each output array is (nstates, nsites).

    """
    nstates = np.prod(state_space_shape)

    edge_to_rate = dict(edge_rate_pairs)
    edge_to_process = dict(edge_process_pairs)

    # For the few nodes that are active at a given point in the traversal,
    # we track a 2d array of shape (nstates, nsites).
    node_to_array = {}
    for node in get_node_evaluation_order(T, root):

        # When a node is activated, its associated array
        # is initialized to its observational likelihood array.
        arr = create_indicator_array(
                node,
                state_space_shape,
                observable_nodes,
                observable_axes,
                iid_observations)

        # Multiplicatively accumulate over outgoing edges.
        for child in T.successors(node):
            edge = (node, child)
            edge_rate = edge_to_rate[edge]
            edge_process = edge_to_process[edge]
            child_arr = node_to_array[child]

            child_edge_arr = f[edge_process].expm_mul(edge_rate, child_arr)
            
            #TODO check this
            #P = f[edge_process].expm_mul(edge_rate, np.identity(nstates))
            #child_edge_arr = child_arr.T.dot(P).T

            arr *= child_edge_arr
            if not store_all:
                del node_to_array[child]

        # Associate the array with the current node.
        node_to_array[node] = arr

    # If we had been deleting arrays as they become unnecessary for
    # the log likelihood calculation, then we would have only
    # a single active array remaining at this point, corresponding to the root.
    # But if we are saving the arrays for gradient calculations,
    # then we have more left.
    actual_keys = set(node_to_array)
    if store_all:
        desired_keys = set(T)
    else:
        desired_keys = {root}
    assert_equal(actual_keys, desired_keys)

    # Return the map from node to array.
    return node_to_array


def get_conditional_likelihoods(
        expm_objects,
        store_all,
        T, root, edges, edge_rate_pairs, edge_process_pairs,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations,
        ):
    """
    Recursively compute conditional likelihoods at the root.

    Attempt to order things intelligently to avoid using
    more memory than is necessary.

    The data provided by the caller gives us a sparse matrix
    of shape (nsites, nnodes, nstates).

    Parameters
    ----------
    expm_objects : sequence of functions indexed by process
        These functions compute expm_mul and rate_mul.
    store_all : bool
        Indicates whether all edge arrays should be stored.

    Returns
    -------
    node_to_conditional_likelihoods : dict
        Maps nodes to ndarrays of shape (nstates, nsites).

    Notes
    -----
    This function computes the likelihood for everything in the subtree
    conditional on the state at the head of the upstream edge, if any,
    of the associated node, without accounting for any observation
    data at that upstream node.
    An alternative representation would track subtree likelihood
    at each node, accounting for the observation data at that node.

    """
    nstates = np.prod(state_space_shape)
    nsites, nobservables = iid_observations.shape

    child_to_edge = dict((tail, (head, tail)) for head, tail in edges)
    edge_to_rate = dict(edge_rate_pairs)
    edge_to_process = dict(edge_process_pairs)

    # For the few nodes that are active at a given point in the traversal,
    # we track a 2d array of shape (nsites, nstates).
    node_to_array = {}
    for node in get_node_evaluation_order(T, root):

        # When a node is activated, its associated array
        # is initialized to its observational likelihood array.
        arr = create_indicator_array(
                node,
                state_space_shape,
                observable_nodes,
                observable_axes,
                iid_observations)

        # When an internal node is activated,
        # this newly activated observational array is elementwise multiplied
        # by each of the active arrays of the child nodes.
        # The new elementwise product becomes the array
        # associated with the activated internal node.
        #
        # If we did not care about saving the per-node arrays,
        # then we could inactivate the child nodes and delete
        # their associated arrays, but because we want to re-use the
        # per-node arrays for edge length gradients, we keep them.
        for child in T.successors(node):
            arr *= node_to_array[child]
            if not store_all:
                del node_to_array[child]

        # When any node that is not the root is activated,
        # the matrix product P.dot(A) replaces A,
        # where A is the active array and P is the matrix exponential
        # associated with the parent edge.
        if node != root:
            edge = child_to_edge[node]
            edge_rate = edge_to_rate[edge]
            edge_process = edge_to_process[edge]
            arr = expm_objects[edge_process].expm_mul(edge_rate, arr)

        # Associate the array with the current node.
        assert_equal(arr.shape, (nstates, nsites))
        node_to_array[node] = arr

    # If we had been deleting arrays as they become unnecessary for
    # the log likelihood calculation, then we would have only
    # a single active array remaining at this point, corresponding to the root.
    # But if we are saving the arrays for gradient calculations,
    # then we have more left.
    actual_keys = set(node_to_array)
    if store_all:
        desired_keys = set(T)
    else:
        desired_keys = {root}
    assert_equal(actual_keys, desired_keys)

    # Return the map from node to array.
    return node_to_array

def get_preorder_conditional_likelihoods(
        expm_objects,
        store_all,
        T, root, edges, edge_rate_pairs, edge_process_pairs,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations,
        prior_distn,
        node_to_postorder_partials,
        ):

    """
    Recursively compute preorder partials at each node.

    Without any attempt to order things intelligently.

    The data provided by the caller gives us a sparse matrix
    of shape (nsites, nnodes, nstates).

    Parameters
    ----------
    expm_objects : sequence of functions indexed by process
        These functions compute expm_mul and rate_mul.
    store_all : bool
        Indicates whether all edge arrays should be stored.

    Returns
    -------
    node_to_preorder_conditional_likelihoods : dict
        Maps nodes to ndarrays of shape (nstates, nsites).

    Notes
    -----
    This function computes the preorder partials for all nodes on the tree.
    """

    nstates = np.prod(state_space_shape)
    nsites, nobservables = iid_observations.shape

    child_to_edge = dict((tail, (head, tail)) for head, tail in edges)
    edge_to_rate = dict(edge_rate_pairs)
    edge_to_process = dict(edge_process_pairs)

    # For the few nodes that are active at a given point in the traversal,
    # we track a 2d array of shape (nsites, nstates).
    node_to_preorder_partials = {}
    for node in list(get_node_evaluation_order(T, root))[::-1]:  # reverse the post-order traversal to get a preorder traversal

        # When a node is activated, its associated array
        # is initialized to its observational likelihood array.
        arr = np.ones((nstates, nsites), dtype=float)

        if node == root:
            arr *= np.transpose([prior_distn])
            node_to_preorder_partials[node] = arr
            continue

        parent_node = list(T.predecessors(node))[0]
        for child in T.successors(parent_node):
            if child != node:
                arr *= node_to_postorder_partials[child]

        arr *= node_to_preorder_partials[parent_node]

        edge = child_to_edge[node]
        edge_rate = edge_to_rate[edge]
        edge_process = edge_to_process[edge]
        arr = expm_objects[edge_process].expm_tmul(edge_rate, arr)

        # Associate the array with the current node.
        assert_equal(arr.shape, (nstates, nsites))
        node_to_preorder_partials[node] = arr

    # If we had been deleting arrays as they become unnecessary for
    # the log likelihood calculation, then we would have only
    # a single active array remaining at this point, corresponding to the root.
    # But if we are saving the arrays for gradient calculations,
    # then we have more left.
    actual_keys = set(node_to_preorder_partials)
    if store_all:
        desired_keys = set(T)
    else:
        desired_keys = {root}
    assert_equal(actual_keys, desired_keys)

    # Return the map from node to array.
    return node_to_preorder_partials
