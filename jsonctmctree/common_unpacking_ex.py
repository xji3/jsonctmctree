"""
Unpack user json input and do some validation.

This should eventually replace common_unpacking.py.

"""
from __future__ import division, print_function, absolute_import

from itertools import product
import re

import numpy as np
import networkx as nx

__all__ = [
        'UnpackingError',
        'TopLevel',
        'interpret_root_prior',
        'interpret_tree',
        'request_regex',
        'gen_valid_extended_properties',
        ]


request_regex = '|'.join((
    '[dsw]nnlogl',
    '[dsw]dnderi',
    '[dsw]dngrad',
    '[dsw][dw][dw]dwel',
    '[dsw][dsw]ntran',
    '[dsw]n[dw]root',
    '[dsw]n[dw]node',
    'ddnance'))


class UnpackingError(Exception):
    pass

class ShapeError(UnpackingError):
    pass

class ContentError(UnpackingError):
    pass


def gen_valid_extended_properties():
    """
    Use the regex to generate all of the valid extended properties.

    """
    core_properties = ('logl', 'deri', 'dwel', 'tran', 'root', 'node')
    for components in product(core_properties, 'dswn', 'dswn', 'dswn'):
        extended_property = ''.join(reversed(components))
        if re.match(request_regex, extended_property):
            yield extended_property


def _unpack(obj, d, f, name):
    """
    Set an attribute of an object.

    Pick a requested key from a provided dictionary, then transform it
    according to the input function and add a member to the object.
    The name of the member should be the key
    and its value should be the transformed value from the dict.

    Parameters
    ----------
    obj : object
        object to be decorated with an attribute
    d : dict
        dict from which the decoration attribute will be taken
    f : function
        requested transform of the value of the attribute
    name : str
        requested attribute name

    """
    try:
        value_in = d[name]
    except KeyError as e:
        raise ContentError('expected attribute "%s"' % name)
    try:
        value_out = f(value_in)
    except UnpackingError as e:
        msg = 'error interpreting attribute "%s" : %s' % (name, str(e))
        raise ContentError(msg)
    return setattr(obj, name, value_out)

def _unpack_object_array(obj, d, f, name):
    try:
        arr_in = d[name]
    except KeyError as e:
        raise ContentError('expected attribute "%s"' % name)
    arr_out = []
    for i, x in enumerate(arr_in):
        try:
            value = f(x)
        except UnpackingError as e:
            raise ContentError('error interpreting entry %d '
                    'of %s: %s' % (i, name, str(e)))
        arr_out.append(value)
    return setattr(obj, name, arr_out)

def _check_ndim(x, desired_ndim=None):
    if desired_ndim is not None:
        actual_ndim = len(x.shape)
        if actual_ndim != desired_ndim:
            raise ShapeError('actual ndim: %d desired ndim: %d' % (
                actual_ndim, desired_ndim))

def _str_lower(x):
    return str(x).lower()

def _np_array(x, dtype=None, ndim=None):
    value = np.array(x, dtype=dtype)
    _check_ndim(value, ndim)
    return value

def _np_array_int_1d(x):
    return _np_array(x, dtype=int, ndim=1)

def _np_array_int_2d(x):
    return _np_array(x, dtype=int, ndim=2)

def _np_array_float_1d(x):
    return _np_array(x, dtype=float, ndim=1)


class TopLevel(object):
    def __init__(self, d):
        _unpack(self, d, Scene, 'scene')
        _unpack_object_array(self, d, Request, 'requests')


class Scene(object):
    def __init__(self, d):
        _unpack(self, d, int, 'node_count')
        _unpack(self, d, int, 'process_count')
        _unpack(self, d, _np_array_int_1d, 'state_space_shape')
        _unpack(self, d, RootPrior, 'root_prior')
        _unpack_object_array(self, d, ProcessDefinition, 'process_definitions')
        _unpack(self, d, Tree, 'tree')
        _unpack(self, d, ObservedData, 'observed_data')

class RootPrior(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_2d, 'states')
        _unpack(self, d, _np_array_float_1d, 'probabilities')
        if self.states.shape[0] != self.probabilities.shape[0]:
            raise ShapeError('in the root prior section of the scene, '
                    'the length of the states array does not match '
                    'that of the probabilities array')

class ProcessDefinition(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_2d, 'row_states')
        _unpack(self, d, _np_array_int_2d, 'column_states')
        _unpack(self, d, _np_array_float_1d, 'transition_rates')
        if self.row_states.shape != self.column_states.shape:
            raise ShapeError('in the process definition section of the scene, '
                    'the shape of the column_states array does not match '
                    'that of the row_states array')
        if self.row_states.shape[0] != self.transition_rates.shape[0]:
            raise ShapeError('in the process definition section of the scene, '
                    'the lengths of the states arrays do not match '
                    'that of the transition_rates array')

class Tree(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_1d, 'row_nodes')
        _unpack(self, d, _np_array_int_1d, 'column_nodes')
        _unpack(self, d, _np_array_float_1d, 'edge_rate_scaling_factors')
        _unpack(self, d, _np_array_int_1d, 'edge_processes')
        if self.row_nodes.shape != self.column_nodes.shape:
            raise ShapeError('in the tree section of the scene, '
                    'the shape of the column_nodes array does not match '
                    'that of the row_nodes array')
        if self.row_nodes.shape != self.edge_rate_scaling_factors.shape:
            raise ShapeError('in the tree section of the scene, '
                    'the shape of the edge_rate_scaling_factors array '
                    'does not match that of the row or column node arrays')
        if self.row_nodes.shape != self.edge_processes.shape:
            raise ShapeError('in the tree section of the scene, '
                    'the shape of the edge_processes array '
                    'does not match that of the row or column node arrays')

class ObservedData(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_1d, 'nodes')
        _unpack(self, d, _np_array_int_1d, 'variables')
        _unpack(self, d, _np_array_int_2d, 'iid_observations')
        if self.nodes.shape != self.variables.shape:
            raise ShapeError('in the observed data section of the scene, '
                    'the shape of the nodes array does not match '
                    'the shape of the variables array')
        if self.nodes.shape[0] != self.iid_observations.shape[1]:
            raise ShapeError('in the observed data section of the scene, '
                    'the length of the nodes array does not match '
                    'the length of each of the iid observation vectors')


class Request(object):
    def __init__(self, d):
        _unpack(self, d, _str_lower, 'property')
        if len(self.property) != 7:
            raise ContentError(
                    'expected the requested property code "%s" '
                    'to consist of seven letters', self.property)
        prefix = self.property[:3]
        suffix = self.property[-4:]
        unrecognized_prefix_letters = set(prefix) - set('dswn')
        if unrecognized_prefix_letters:
            raise ContentError(
                    'expected the first three letters of the requested '
                    'property code "%s" to each belong to {d, s, w, n}' % (
                        self.property))
        if not re.match(request_regex, self.property):
            raise ContentError(
                    'the requested property code "%s" '
                    'is correctly formatted but does not correspond '
                    'to a supported property' % self.property)
        transition_code = 'w' if suffix == 'tran' else 'n'
        actual_keys = set(d)
        desired_keys = {'property'}
        name_reduction_pairs = (
                ('observation', ObservationReduction),
                ('edge', EdgeReduction),
                ('state', StateReduction),
                ('transition', TransitionReduction),
                )
        names = ('observation', 'edge', 'state', 'transition')
        for c, pair in zip(prefix + transition_code, name_reduction_pairs):
            name, reduction = pair
            if c == 'w':
                key = name + '_reduction'
                desired_keys.add(key)
                _unpack(self, d, reduction, key)
        if actual_keys != desired_keys:
            raise ContentError('The members of a request object do not '
                    'correspond to those expected '
                    'given its property code "%s"; '
                    'actual set of members: %s desired set of members: %s' % (
                        self.property, actual_keys, desired_keys))

class ObservationReduction(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_1d, 'observation_indices')
        _unpack(self, d, _np_array_float_1d, 'weights')
        if self.observation_indices.shape != self.weights.shape:
            raise ShapeError('in a requested observation reduction, '
                    'expected the shape of the observation_indices array '
                    'to be equal to the shape of the weights array')

class EdgeReduction(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_1d, 'edges')
        _unpack(self, d, _np_array_float_1d, 'weights')
        if self.edges.shape != self.weights.shape:
            raise ShapeError('in a requested edge reduction, '
                    'expected the shape of the edges array '
                    'to be equal to the shape of the weights array')

class StateReduction(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_2d, 'states')
        _unpack(self, d, _np_array_float_1d, 'weights')
        if self.states.shape[0] != self.weights.shape[0]:
            raise ShapeError('in a requested state reduction, '
                    'expected the length of the states array '
                    'to be equal to the length of the weights array')

class TransitionReduction(object):
    def __init__(self, d):
        _unpack(self, d, _np_array_int_2d, 'row_states')
        _unpack(self, d, _np_array_int_2d, 'column_states')
        _unpack(self, d, _np_array_float_1d, 'weights')
        if self.row_states.shape != self.column_states.shape:
            raise ShapeError('in a requested transition reduction, '
                    'expected the shape of the row states array '
                    'to be equal to the shape of the column states array')
        if self.row_states.shape[0] != self.weights.shape[0]:
            raise ShapeError('in the requested state reduction, '
                    'expected the lengths of the states arrays '
                    'to be equal to the length of the weights array')


#####################################################
# interpretations
# This is an additional layer of processing
# after the json parsing and its types conversion.


def interpret_root_prior(scene):
    # Interpret the prior distribution by converting it to a dense array.
    nstates = np.prod(scene.state_space_shape)
    feas = np.ravel_multi_index(
            scene.root_prior.states.T,
            scene.state_space_shape)
    distn = np.zeros(nstates, dtype=float)
    np.put(distn, feas, scene.root_prior.probabilities)
    return distn


def _check_tree_row_indices(row, node_count):
    """This is just for error checking.
    """
    nodes = range(node_count)
    unexpected_row_indices = list(set(row) - set(nodes))
    if unexpected_row_indices:
        raise ContentError(
                'Found unexpected row indices in the tree definition. '
                'Because the provided node count is %d, '
                'the row indices are expected to be non-negative integers '
                'less than %d. '
                'But the following row indices were observed: %s' % (
                    node_count, node_count, unexpected_row_indices))


def _check_tree_col_indices(col, node_count):
    """This is just for error checking.
    """
    nodes = range(node_count)
    unexpected_col_indices = list(set(col) - set(nodes))
    if unexpected_col_indices:
        raise ContentError(
                'Found unexpected col indices in the tree definition. '
                'Because the provided node count is %d, '
                'the col indices are expected to be non-negative integers '
                'less than %d. '
                'But the following col indices were observed: %s' % (
                    node_count, node_count, unexpected_col_indices))

def interpret_tree(scene):
    nodes = range(scene.node_count)
    _check_tree_row_indices(scene.tree.row_nodes, scene.node_count)
    _check_tree_col_indices(scene.tree.column_nodes, scene.node_count)
    if np.min(scene.tree.edge_rate_scaling_factors) < 0:
        raise ContentError(
                'the edge-specific rate scaling factors '
                'should be non-negative')
    T = nx.DiGraph()
    T.add_nodes_from(nodes)
    edges = list(zip(scene.tree.row_nodes, scene.tree.column_nodes))
    T.add_edges_from(edges)
    if len(T.edges()) != len(edges):
        raise ContentError('the tree has an unexpected number of edges')
    if len(edges) + 1 != len(T):
        raise ContentError('expected the number of edges to be one more '
                'than the number of nodes')
    in_degree = T.in_degree()
    roots = [n for n in nodes if in_degree[n] == 0]
    if len(roots) != 1:
        raise ContentError('expected exactly one root')
    for i in range(scene.node_count):
        T.in_degree()
    root = roots[0]
    edges = list(zip(scene.tree.row_nodes, scene.tree.column_nodes))
    edge_rate_pairs = list(zip(edges, scene.tree.edge_rate_scaling_factors))
    edge_process_pairs = list(zip(edges, scene.tree.edge_processes))
    return T, root, edges, edge_rate_pairs, edge_process_pairs
