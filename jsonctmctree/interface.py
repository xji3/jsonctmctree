"""
Begin a new interface.

This includes support for user requests for six 'posterior' base properties:
    * LOGL: log likelihood
    * DERI: derivatives with respect to log edge rates
    * TRAN: transition count expectations
    * DWEL: dwell proportion expectations
    * ROOT: state count expectations at the root
    * NODE: state count expectations at all nodes

Each base property is extended to allow one or more reductions:
    * reduction across iid observations (observation_reduction)
    * reduction across edges (edge_reduction)
    * reduction across states (state_reduction)

Each requested reduction may be defined by either simple summation
or by a user-specified weighted sum over a user-specified sequence of indices.

The transition base property request always requires an additional
weighted reduction (transition_reduction) that is defined on multivariate
state pairs.

For each property request,
the response array will have one axis for each D in the property prefix.
If the array has zero axes then a single floating point number will be returned.

The extended properties have 7-letter names according to the following scheme:
observation (1) | edge (1) | state (1) | base property name (4)
where the letters of the 3-letter prefix are from:
(D)istinct (no reduction)
(S)um (unweighted sum)
(W)eighted sum
(N)ot applicable

The 6 base properties can be extended as follows to a total of 39 properties:
{D,S,W}NNLOGL : 3
{D,S,W}DNDERI : 3
{D,S,W}{D,W}{D,W}DWEL : 12
{D,S,W}{D,S,W}NTRAN : 9
{D,S,W}N{D,W}ROOT : 6
{D,S,W}N{D,W}NODE : 6

The interface is limited in that it does not support the following:
    * continuous observations along time intervals
    * non-axis-aligned state aggregate observations
    * noisy observations (subsumes state aggregate observations)
    * partitions of observations within a single scene
    * second derivatives and cross derivatives
    * uncertainty in the branching structure of the timeline
    * random effects
    * inference and hypothesis testing are not performed automatically

"""
from __future__ import division, print_function, absolute_import

from . import impl_naive, impl_v2
from .pyexp import _expm_multiply


def diagnose_expm_multiply():
    return _expm_multiply.expm_multiply(None, None)


def process_json_in(j_in, debug=False, seed = None):
    """
    The part of the input that is the same across requests is as follows.
    I'm bundling all of this stuff together and calling it a 'scene'.

    'scene' : {
        'node_count' : 4,
        'process_count' : 2,
        'state_space_shape' : [2, 2],
        'root_prior' : {
            'states' : ...,
            'probabilities' : ...},
        'tree' : {
            'row_nodes' : ...,
            'column_nodes' : ...,
            'edge_rate_scaling_factors' : ...,
            'edge_processes' : ...},
        'process_definitions' : [
            {
                'row_states' : ...,
                'column_states' : ...,
                'transition_rates' : ...}, ...],
        'observed_data' : {
            'nodes' : [...],
            'variables' : [...],
            'iid_observations' : [[...]]}
    }

    The requests part of the input is an array of json objects,
    each of which has a 'property' (one of the 39 properties listed above)
    and may have one or more weighted reduction members.
    The number of weighted reduction definition members is equal to the
    number of 'w' characters in the 3-letter prefix of the property.
    Each weighted reduction definition is an object with an array
    of indices and an array of weights.

    'requests' : [
        {
            'property' : 'snnlogl'
        },
        {
            'property' : 'wwwdwel',
            'observation_reduction' : {
                'observation_indices' : [1, 1, 1],
                'weights' : [1, 1, 1]},
            'edge_reduction' : {
                'edges' : [1, 1, 1],
                'weights' : [1, 1, 1]},
            'state_reduction' : {
                'states' : [[0, 0], [0, 1], [0, 2]],
                'weights' : [1, 1, 1]}
        },
        {
            'property' : 'wsntran',
            'observation_reduction' : {
                'observation_indices' : [1, 1, 1],
                'weights' : [1, 1, 1]},
            'transition_reduction' : {
                'row_states' : [[0, 0], [0, 1], [0, 2]],
                'column_states' : [[0, 1], [0, 2], [0, 0]],
                'weights' : [1, 1, 1]}
        }
        ]

    """
    return impl_v2.process_json_in(j_in, debug=debug, seed = seed)
