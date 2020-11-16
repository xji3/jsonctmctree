"""
Implement log likelihoods for complicated models.

This interface takes some care about memory usage,
while allowing more subtlety in the representation of observed data,
and while allowing more flexibility in the representation of
inhomogeneity of the process across branches.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np
from numpy.testing import assert_equal

from .expm_helpers import PadeExpm, EigenExpm, ActionExpm

from .common_unpacking import (
        SimpleError,
        SimpleShapeError,
        get_observables_info,
        get_tree_info,
        get_prior_info)

from .common_likelihood import (
        create_indicator_array,
        get_conditional_likelihoods)


def get_site_weights(j_in):
    return np.array(j_in['site_weights'])


def get_requested_derivatives(j_in):
    return np.array(j_in['requested_derivatives'])


def get_processes_info(j_in):
    processes_row = []
    processes_col = []
    processes_rate = []
    for j_process in j_in['processes']:
        processes_row.append(j_process['row'])
        processes_col.append(j_process['col'])
        processes_rate.append(j_process['rate'])
    return (
            np.array(processes_row),
            np.array(processes_col),
            np.array(processes_rate, dtype=float))


def get_edge_derivative(
        # functions to compute expm_mul and rate_mul, per process (NEW),
        f,
        # single derivative edge (pair of head, tail nodes not index) (NEW)
        derivative_edge,
        # map from node to array returned by (NEW)
        node_to_array,
        # original args
        T, root, edges, edge_rate_pairs, edge_process_pairs,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations,
        ):
    """

    """
    # Some preprocessing.
    child_to_edge = dict((tail, (head, tail)) for head, tail in edges)
    edge_to_rate = dict(edge_rate_pairs)
    edge_to_process = dict(edge_process_pairs)

    # Unpack the edge of interest.
    derivative_head_node, derivative_tail_node = derivative_edge

    # For each edge use a new edge-specific array.
    # Trace the evaluation back to the root.
    node_to_deriv_array = {}

    # Iterate over nodes on the path from the derivative edge tail node
    # up to the root of the tree.
    node = derivative_tail_node
    while True:

        # When a node is activated, its associated array
        # is initialized to its observational likelihood array.
        arr = create_indicator_array(
                node,
                state_space_shape,
                observable_nodes,
                observable_axes,
                iid_observations)

        # If we are not analyzing the root node then determine
        # the characteristics of the edge upstream of the node.
        if node != root:
            edge = child_to_edge[node]
            edge_rate = edge_to_rate[edge]
            edge_process = edge_to_process[edge]

        # At the tail node, compute the adjusted array
        # for the edge of interest.
        # At non-tail nodes, including the root,
        # compute array using the same method as in the derivative-free
        # evaluation.
        # Arrays that are specific to the edge of interest
        # may be deleted after they are used.
        if node == derivative_tail_node:
            arr = node_to_array[node]
            arr = f[edge_process].rate_mul(edge_rate, arr)
        else:
            for child in T.successors(node):
                if child in node_to_deriv_array:
                    arr *= node_to_deriv_array[child]
                    del node_to_deriv_array[child]
                else:
                    arr *= node_to_array[child]
            if node != root:
                arr = f[edge_process].expm_mul(edge_rate, arr)

        # Associate the array with the current node.
        node_to_deriv_array[node] = arr

        # If the current node is the root node then we are done.
        # Otherwise move to the parent node.
        if node == root:
            break
        else:
            head_node, tail_node = edge
            node = head_node

    # Among the arrays specific to the analysis of the derivative
    # of the edge of interest, only the array at the root
    # should still be active at this point.
    assert_equal(set(node_to_deriv_array), {root})
    return node_to_deriv_array[root]


def get_edge_derivatives(
        f,
        requested_derivative_edge_indices,
        node_to_array,
        distn,
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
    f : functions to compute expm_mul and rate_mul, per process
        functions
    requested_derivative_edge_indices : set of edge indices
        set of requested edge indices for edge rate derivatives
    node_to_array : dict
        map from node to array returned by get_conditional_likelihoods
    distn : 1d array
        prior state distribution at the root

    """
    child_to_edge = dict((tail, (head, tail)) for head, tail in edges)
    edge_to_rate = dict(edge_rate_pairs)
    edge_to_process = dict(edge_process_pairs)

    # Compute the likelihood derivative for each requested edge length.
    edge_index_to_derivatives = dict()
    for edge_index in requested_derivative_edge_indices:

        # Get the edge that corresponds to the edge index of interest.
        derivative_edge = edges[edge_index]

        # Compute the array at the root node.
        arr = get_edge_derivative(
                f,
                derivative_edge,
                node_to_array,
                T, root, edges, edge_rate_pairs, edge_process_pairs,
                state_space_shape,
                observable_nodes,
                observable_axes,
                iid_observations)

        # Apply the prior distribution to the array.
        # Now we have the derivatives of the likelihoods with respect
        # to the edge-specific rate scaling factors.
        # We want the derivatives of the log likelihoods with respect
        # to the log of the edge-specific rate scaling factors.
        derivatives = distn.dot(arr)
        edge_index_to_derivatives[edge_index] = derivatives

    # Return the map from edge index to edge-specific derivatives.
    return edge_index_to_derivatives


def get_edge_gradients(
        f,
        requested_derivative_edge_indices,
        node_to_postorder_array,
        node_to_preorder_array,
        T, root, edges, edge_rate_pairs, edge_process_pairs,
        state_space_shape,
        observable_nodes,
        observable_axes,
        iid_observations,
        ):
    """
    Recursively compute gradients at each edge.

    Parameters
    ----------
    f : functions to compute expm_mul and rate_mul, per process
        functions
    requested_derivative_edge_indices : set of edge indices
        set of requested edge indices for edge rate derivatives
    node_to_postorder_array : dict
        map from node to array returned by get_conditional_likelihoods
    distn : 1d array
        prior state distribution at the root

    """
    child_to_edge = dict((tail, (head, tail)) for head, tail in edges)
    edge_to_rate = dict(edge_rate_pairs)
    edge_to_process = dict(edge_process_pairs)

    # Compute the likelihood derivative for each requested edge length.
    edge_index_to_derivatives = dict()
    for edge_index in requested_derivative_edge_indices:

        # Get the edge that corresponds to the edge index of interest.
        derivative_edge = edges[edge_index]

        preorder_partial = node_to_preorder_array[derivative_edge[1]]
        postorder_partial = create_indicator_array(
                derivative_edge[1],
                state_space_shape,
                observable_nodes,
                observable_axes,
                iid_observations)
        for child in T.successors(derivative_edge[1]):
            postorder_partial *= node_to_postorder_array[child]

        # Compute the array at the root node.
        edge_process = edge_to_process[derivative_edge]
        gradient = f[edge_process].gradient_red(postorder_partial, preorder_partial)

        edge_index_to_derivatives[edge_index] = gradient

    # Return the map from edge index to edge-specific derivatives.
    return edge_index_to_derivatives

def process_json_in(j_in):

    # Unpack some sizes and shapes.
    nnodes = j_in['node_count']
    nprocesses = j_in['process_count']
    state_space_shape = np.array(j_in['state_space_shape'])

    # Unpack stuff related to the tree and its edges.
    info = get_tree_info(j_in)
    T, root, edges, edge_rate_pairs, edge_process_pairs = info

    # Unpack site weights. (NEW)
    site_weights = get_site_weights(j_in)

    # Unpack the indices of the edges for which the derivatives
    # of the likelihood are requested. (NEW)
    requested_derivatives = get_requested_derivatives(j_in)

    # Unpack stuff related to observables.
    info = get_observables_info(j_in, nnodes, state_space_shape)
    observable_nodes, observable_axes, iid_observations = info

    # Unpack stuff related to the prior distribution.
    info = get_prior_info(j_in)
    prior_feasible_states, prior_distribution = info

    # Unpack stuff related to the edge-specific processes.
    info = get_processes_info(j_in)
    processes_row, processes_col, processes_rate = info

    # Interpret the prior distribution.
    nstates = np.prod(state_space_shape)
    feas = np.ravel_multi_index(prior_feasible_states.T, state_space_shape)
    distn = np.zeros(nstates, dtype=float)
    np.put(distn, feas, prior_distribution)

    # For each process, precompute the objects that are capable
    # of computing expm_mul and rate_mul for log likelihoods
    # and for its derivative with respect to edge-specific rates.
    #expm_klass = EigenExpm # TODO soft-code this
    #expm_klass = PadeExpm # TODO soft-code this
    expm_klass = ActionExpm # TODO soft-code this
    f = []
    for edge_process in range(nprocesses):
        row = processes_row[edge_process]
        col = processes_col[edge_process]
        rate = processes_rate[edge_process]
        obj = expm_klass(state_space_shape, row, col, rate)
        f.append(obj)

    # Determine whether to store intermediate arrays
    # or whether to store only as many as necessary to compute
    # the log likelihood.
    store_all_likelihood_arrays = bool(np.any(requested_derivatives))

    # Precompute conditional likelihood arrays per node.
    node_to_array = get_conditional_likelihoods(
            f, store_all_likelihood_arrays,
            T, root, edges, edge_rate_pairs, edge_process_pairs,
            state_space_shape,
            observable_nodes,
            observable_axes,
            iid_observations)

    # Get likelihoods at the root.
    # These are passed to the derivatives procedure,
    # to help compute the per-site derivatives of the log likelihoods
    # with respect to the log of the edge-specific rate scaling parameters.
    arr = node_to_array[root]
    likelihoods = distn.dot(arr)
    
    # Compute the derivative of the likelihood
    # with respect to each edge-specific rate scaling parameter.
    requested_derivative_edge_indices = set(requested_derivatives)
    ei_to_derivatives = get_edge_derivatives(
            f, requested_derivative_edge_indices,
            node_to_array, distn,
            T, root, edges, edge_rate_pairs, edge_process_pairs,
            state_space_shape,
            observable_nodes,
            observable_axes,
            iid_observations)

    # Apply the prior distribution and take logs of the likelihoods.
    log_likelihoods = np.log(likelihoods)

    # Adjust for infeasibility.
    feasibilities = np.isfinite(log_likelihoods)
    log_likelihoods = np.where(feasibilities, log_likelihoods, 0)

    # Reduce the log likelihoods according to the site weights.
    if np.all(feasibilities):
        feasibility = True
        log_likelihood = np.dot(site_weights, log_likelihoods)

        # Process the derivatives.
        # They are currently in the form of derivatives of edge rates
        # with respect to the likelihood, but we want to convert them to
        # derivatives of log likelihood with respect to the logs
        # of edge-specific rate scaling factors.
        # According to calculus we can do it as follows.
        # Also reduce the derivative array according to the site weights.
        edge_to_rate = dict(edge_rate_pairs)
        ei_to_d = {}
        for ei, derivatives in ei_to_derivatives.items():
            edge = edges[ei]
            edge_rate = edge_to_rate[edge]
            d = site_weights.dot(derivatives / likelihoods)
            ei_to_d[ei] = float(d)

        # Map the derivatives back to a list whose entries
        # match the requested order of the indices.
        derivatives_out = [ei_to_d[ei] for ei in requested_derivatives]

    else:
        feasibility = False
        log_likelihood = 0
        derivatives_out = [0] * len(requested_derivatives)


    # Create the output in a format that json will like.
    j_out = dict(
            status = 'success',
            feasibility = feasibility,
            log_likelihood = log_likelihood,
            edge_derivatives = derivatives_out)

    return j_out
