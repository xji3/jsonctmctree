"""
A more advanced implementation of interface.py.

This implementation is more complicated and should be more efficient
than the naive implementation.

"""
from __future__ import division, print_function, absolute_import

import sys

import numpy as np
from numpy.testing import assert_equal, assert_

from .expm_helpers import (
        ActionExpm,
        ImplicitDwellExpmFrechet,
        ImplicitTransitionExpmFrechetEx,
        )
from .common_likelihood import (
        get_conditional_likelihoods, get_subtree_likelihoods, get_preorder_conditional_likelihoods)
from .common_unpacking_ex import TopLevel, interpret_tree, interpret_root_prior
from .common_reduction import apply_prefixed_reductions, apply_reductions
from . import expect
from . import ll
from .impl_naive import (
        _eagerly_precompute_dwell_objects,
        _apply_eagerly_precomputed_dwell_objects,
        _compute_transition_expectations,
        )


class InfeasibilityError(Exception):
    pass


class Reactor(object):
    """
    This is like a state machine.

    """
    def __init__(self, scene, debug=False):
        self.scene = scene
        self.debug = debug
        # interpret some stuff
        self.prior_distn = interpret_root_prior(scene)
        (
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                ) = interpret_tree(scene)
        # init arrays
        self.checked_feasibility = False
        self.node_to_subtree_likelihoods = None
        self.node_to_conditional_likelihoods = None
        self.node_to_preorder_conditional_likelihoods = None
        self.node_to_marginal_distn = None
        self.likelihoods = None
        self.log_likelihoods = None
        self.derivatives = None
        self.gradients = None
        # If only the likelihoods and log likelihoods are required,
        # for example to check feasibility or to return log likelihoods
        # or to compute the posterior distribution at the root,
        # then only the root conditional likelihoods are required.
        self.root_conditional_likelihoods = None
        # The root marginal distribution can be computed using
        # the root conditional likelihoods or node_to_subtree_likelihoods
        # or node_to_conditional_likelihoods.
        self.root_marginal_distn = None

        # For each process, precompute the objects that are capable
        # of computing expm_mul and rate_mul for log likelihoods
        # and for its derivative with respect to edge-specific rates.
        self.expm_objects = []
        for p in scene.process_definitions:
            obj = ActionExpm(
                    scene.state_space_shape,
                    p.row_states,
                    p.column_states,
                    p.transition_rates,
                    debug=debug)
            self.expm_objects.append(obj)
        self._note('reactor is initialized')

    def _note(self, msg):
        if self.debug:
            print(msg, file=sys.stderr)

    def _delete_root_marginal_distn(self, unmet_core_requests):
        if self.root_marginal_distn is None:
            return False
        if unmet_core_requests & {'root'}:
            return False
        self.root_marginal_distn = None
        return True

    def _delete_root_conditional_likelihoods(self, unmet_core_requests):
        if self.root_conditional_likelihoods is None:
            return False
        if not self.checked_feasibility:
            return False
        if unmet_core_requests & {'logl', 'deri', 'grad', 'root'}:
            return False
        self.root_conditional_likelihoods = None
        return True

    def _delete_likelihoods(self, unmet_core_requests):
        if self.likelihoods is None:
            return False
        if not self.checked_feasibility:
            return False
        if unmet_core_requests & {'deri', 'grad'}:
            return False
        if unmet_core_requests & {'logl'}:
            if self.log_likelihoods is None:
                return False
        self.likelihoods = None
        return True

    def _delete_log_likelihoods(self, unmet_core_requests):
        if self.log_likelihoods is None:
            return False
        if unmet_core_requests & {'logl'}:
            return False
        self.log_likelihoods = None
        return True

    def _delete_node_to_subtree_likelihoods(self, unmet_core_requests):
        if self.node_to_subtree_likelihoods is None:
            return False
        if unmet_core_requests & {'dwel', 'tran', 'root', 'node'}:
            return False
        self.node_to_subtree_likelihoods = None
        return True

    def _delete_node_to_conditional_likelihoods(self, unmet_core_requests):
        if self.node_to_conditional_likelihoods is None:
            return False
        if unmet_core_requests & {'logl', 'deri', 'grad', 'root'}:
            return False
        self.node_to_conditional_likelihoods = None

        if unmet_core_requests & {'grad'}:
            self.node_to_preorder_conditional_likelihoods = None
        return True

    def _delete_node_to_marginal_distn(self, unmet_core_requests):
        if self.node_to_marginal_distn is None:
            return False
        if unmet_core_requests & {'dwel', 'tran', 'root', 'node'}:
            return False
        self.node_to_marginal_distn = None
        return True

    def _delete_derivatives(self, unmet_core_requests):
        if self.derivatives is None:
            return False
        if unmet_core_requests & {'deri', 'grad'}:
            return False
        self.derivatives = None
        return True

    def _delete_gradients(self, unmet_core_requests):
        if self.gradients is None:
            return False
        if unmet_core_requests & {'grad'}:
            return False
        self.gradients = None
        return True


    def _check_feasibility(self):
        if self.checked_feasibility:
            return False
        if self.likelihoods is None:
            return False
        self.checked_feasibility = True
        if not np.all(self.likelihoods):
            raise InfeasibilityError
        return True


    def _create_root_marginal_distn(self, unmet_core_requests):
        if self.root_marginal_distn is not None:
            return False
        if not (unmet_core_requests & {'root'}):
            return False
        if self.node_to_marginal_distn is not None:
            self.root_marginal_distn = self.node_to_marginal_distn[self.root]
            return True
        elif self.root_conditional_likelihoods is not None:
            root_arr = self.root_conditional_likelihoods
        elif self.node_to_subtree_likelihoods is not None:
            root_arr = self.node_to_subtree_likelihoods[self.root]
        elif self.node_to_conditional_likelihoods is not None:
            root_arr = self.node_to_conditional_likelihoods[self.root]
        else:
            return False
        full_array = root_arr * self.prior_distn[:, np.newaxis]
        col_sums_recip = expect.pseudo_reciprocal(full_array.sum(axis=0))
        self.root_marginal_distn = full_array * col_sums_recip
        return True

    def _create_likelihoods(self, unmet_core_requests):
        if self.likelihoods is not None:
            return False
        if self.checked_feasibility:
            # likelihoods are required for feasibility checking...
            return False
        if self.root_conditional_likelihoods is not None:
            arr = self.root_conditional_likelihoods
        elif self.node_to_subtree_likelihoods is not None:
            arr = self.node_to_subtree_likelihoods[self.root]
        elif self.node_to_conditional_likelihoods is not None:
            arr = self.node_to_conditional_likelihoods[self.root]
        else:
            return False
        self.likelihoods = self.prior_distn.dot(arr)
        assert_equal(len(self.likelihoods.shape), 1)
        return True

    def _create_log_likelihoods(self, unmet_core_requests):
        if self.log_likelihoods is not None:
            return False
        if not (unmet_core_requests & {'logl'}):
            return False
        if self.likelihoods is None:
            return False
        self.log_likelihoods = np.log(self.likelihoods)
        return True

    def _create_derivatives(self, unmet_core_requests):
        if self.derivatives is not None:
            return False
        if not (unmet_core_requests & {'deri'}):
            return False
        if self.likelihoods is None:
            return False
        if self.node_to_conditional_likelihoods is None:
            return False

        # Compute the derivative of the likelihood
        # with respect to each edge-specific rate scaling parameter.
        #TODO do not necessarily request all edge derivatives
        nedges = len(self.edges)
        requested_derivative_edge_indices = set(range(nedges))
        ei_to_derivatives = ll.get_edge_derivatives(
                self.expm_objects, requested_derivative_edge_indices,
                self.node_to_conditional_likelihoods, self.prior_distn,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                self.scene.state_space_shape,
                self.scene.observed_data.nodes,
                self.scene.observed_data.variables,
                self.scene.observed_data.iid_observations)

        # Fill an array with all unreduced derivatives.
        iid_observation_count = len(self.scene.observed_data.iid_observations)
        self.derivatives = np.empty((iid_observation_count, nedges))
        for ei, der in ei_to_derivatives.items():
            self.derivatives[:, ei] = der / self.likelihoods
        return True

    def _create_gradients(self, unmet_core_requests):
        if self.derivatives is not None:
            return False
        if not (unmet_core_requests & {'grad'}):
            return False
        if self.likelihoods is None:
            return False
        if self.node_to_conditional_likelihoods is None:
            return False

        # Compute the derivative of the likelihood
        # with respect to each edge-specific rate scaling parameter.
        #TODO do not necessarily request all edge derivatives
        nedges = len(self.edges)
        requested_derivative_edge_indices = set(range(nedges))
        ei_to_derivatives = ll.get_edge_derivatives(
                self.expm_objects, requested_derivative_edge_indices,
                self.node_to_conditional_likelihoods, self.prior_distn,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                self.scene.state_space_shape,
                self.scene.observed_data.nodes,
                self.scene.observed_data.variables,
                self.scene.observed_data.iid_observations)

        # Fill an array with all unreduced derivatives.
        iid_observation_count = len(self.scene.observed_data.iid_observations)
        self.derivatives = np.empty((iid_observation_count, nedges))
        for ei, der in ei_to_derivatives.items():
            self.derivatives[:, ei] = der / self.likelihoods
        return True

    def _create_root_conditional_likelihoods(self, unmet_core_requests):
        if self.root_conditional_likelihoods is not None:
            return False
        if self.node_to_subtree_likelihoods is not None:
            return False
        if self.node_to_conditional_likelihoods is not None:
            return False
        if unmet_core_requests & {'deri', 'grad'}:
            # in this case we need all conditional likelihoods not just root
            return False
        if unmet_core_requests & {'dwel', 'trans', 'node'}:
            # in these cases we need all subtree likelihoods not just root
            return False
        if self.checked_feasibility:
            if not (unmet_core_requests & {'logl', 'root'}):
                return False
        store_all = False
        d = get_conditional_likelihoods(
                self.expm_objects,
                store_all,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                self.scene.state_space_shape,
                self.scene.observed_data.nodes,
                self.scene.observed_data.variables,
                self.prior_distn,
                )
        self.root_conditional_likelihoods = d[self.root]
        return True

    def _create_node_to_conditional_likelihoods(self, unmet_core_requests):
        if self.node_to_conditional_likelihoods is not None:
            return False
        if not (unmet_core_requests & {'deri', 'grad'}):
            # other likelihood objects can be used for non-deri applications
            return False
        store_all = True
        self.node_to_conditional_likelihoods = get_conditional_likelihoods(
                self.expm_objects,
                store_all,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                self.scene.state_space_shape,
                self.scene.observed_data.nodes,
                self.scene.observed_data.variables,
                self.scene.observed_data.iid_observations,
                )

        if (unmet_core_requests & {'grad'}):
            # store preorder partials too
            self.node_to_preorder_conditional_likelihoods = get_preorder_conditional_likelihoods(
                self.expm_objects,
                store_all,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                self.scene.state_space_shape,
                self.scene.observed_data.nodes,
                self.scene.observed_data.variables,
                self.scene.observed_data.iid_observations,
                self.prior_distn,
            )
        return True

    def _create_node_to_subtree_likelihoods(self, unmet_core_requests):
        if self.node_to_subtree_likelihoods is not None:
            return False
        if not (unmet_core_requests & {'dwel', 'tran', 'node'}):
            # other likelihood objects can take over for other applications
            return False
        #TODO restrict the requested number of arrays
        store_all = True
        self.node_to_subtree_likelihoods = get_subtree_likelihoods(
                self.expm_objects,
                store_all,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                self.scene.state_space_shape,
                self.scene.observed_data.nodes,
                self.scene.observed_data.variables,
                self.scene.observed_data.iid_observations,
                )
        return True

    def _create_node_to_marginal_distn(self, unmet_core_requests):
        if self.node_to_marginal_distn is not None:
            return False
        if not (unmet_core_requests & {'dwel', 'tran', 'node'}):
            return False
        if self.node_to_subtree_likelihoods is None:
            return False
        debug = False
        self.node_to_marginal_distn = expect.get_node_to_marginal_distn(
                self.expm_objects,
                self.node_to_subtree_likelihoods, self.prior_distn,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                self.scene.state_space_shape,
                self.scene.observed_data.nodes,
                self.scene.observed_data.variables,
                self.scene.observed_data.iid_observations,
                debug=debug)
        return True


    #FIXME
    # site, edge, state
    #{D,S,W}NNLOGL : 3
    #{D,S,W}DNDERI : 3
    #{D,S,W}{D,W}{D,W}DWEL : 12
    #{D,S,W}{D,S,W}NTRAN : 9
    #{D,S,W}N{D,W}ROOT : 6
    #{D,S,W}N{D,W}NODE : 6

    def _respond_to_root(self, unmet_core_requests, requests, responses):
        if 'root' not in unmet_core_requests:
            return False
        if self.root_marginal_distn is None:
            return False
        full_array = self.root_marginal_distn.T
        for i, request in enumerate(requests):
            prefix = request.property[:3]
            suffix = request.property[-4:]
            if suffix == 'root':
                s = self.scene.state_space_shape
                out = apply_reductions(s, request, full_array)
                responses[i] = out.tolist()
        return True

    def _respond_to_logl(self, unmet_core_requests, requests, responses):
        if 'logl' not in unmet_core_requests:
            return False
        if self.log_likelihoods is None:
            return False
        for i, request in enumerate(requests):
            prefix = request.property[:3]
            suffix = request.property[-4:]
            if suffix == 'logl':
                s = self.scene.state_space_shape
                out = apply_reductions(s, request, self.log_likelihoods)
                responses[i] = out.tolist()
        return True

    def _respond_to_deri(self, unmet_core_requests, requests, responses):
        if 'deri' not in unmet_core_requests:
            return False
        if self.derivatives is None:
            return False
        for i, request in enumerate(requests):
            prefix = request.property[:3]
            suffix = request.property[-4:]
            if suffix == 'deri':
                s = self.scene.state_space_shape
                out = apply_reductions(s, request, self.derivatives)
                responses[i] = out.tolist()
        return True

    def _respond_to_grad(self, unmet_core_requests, requests, responses):
        if 'grad' not in unmet_core_requests:
            return False
        if self.gradients is None:
            return False
        for i, request in enumerate(requests):
            prefix = request.property[:3]
            suffix = request.property[-4:]
            if suffix == 'grad':
                s = self.scene.state_space_shape
                out = apply_reductions(s, request, self.derivatives)
                responses[i] = out.tolist()
        return True

    def _respond_to_node(self, unmet_core_requests, requests, responses):
        if 'node' not in unmet_core_requests:
            return False
        if self.node_to_marginal_distn is None:
            return False
        arr = []
        for node in range(self.scene.node_count):
            arr.append(self.node_to_marginal_distn[node])
        full_node_array = np.array(arr).T
        for i, request in enumerate(requests):
            prefix = request.property[:3]
            suffix = request.property[-4:]
            if suffix == 'node':
                s = self.scene.state_space_shape
                out = apply_reductions(s, request, full_node_array)
                responses[i] = out.tolist()
        return True

    def _respond_to_dwel(self, unmet_core_requests, requests, responses):
        if 'dwel' not in unmet_core_requests:
            return False
        if self.node_to_subtree_likelihoods is None:
            return False
        if self.node_to_marginal_distn is None:
            return False

        # Precompute some counts.
        nsites = self.scene.observed_data.iid_observations.shape[0]
        nstates = np.prod(self.scene.state_space_shape)

        # If any dwell request does not reduce the 'state' axis,
        # or if the number of states is less than the number
        # of 'dwel' requests, then precompute all dwell objects up front
        # and use reduction to meet each request.
        # Otherwise create request-specific dwell objects.
        precompute_per_state = False
        unmet_dwel_request_count = 0
        for request, response in zip(requests, responses):
            if response is not None:
                continue
            suffix = request.property[-4:]
            if suffix != 'dwel':
                continue
            prefix = request.property[:3]
            if prefix[2] == 'd':
                precompute_per_state = True
            unmet_dwel_request_count += 1
        if nstates <= unmet_dwel_request_count:
            precompute_per_state = True

        if precompute_per_state:
            # Apply the precomputed dwell objects, creating
            # an array like (nsites, nedges, nstates) after the transposition.
            all_dwell_objects = _eagerly_precompute_dwell_objects(self.scene)
            arr = []
            for dwell_state_index in range(nstates):
                dwell_objects = all_dwell_objects[dwell_state_index]
                arr.append(_apply_eagerly_precomputed_dwell_objects(
                        self.scene,
                        self.expm_objects,
                        dwell_objects,
                        self.node_to_marginal_distn,
                        self.node_to_subtree_likelihoods,
                        self.prior_distn,
                        self.T,
                        self.root,
                        self.edges,
                        self.edge_rate_pairs,
                        self.edge_process_pairs))
            full_dwell_array = np.array(arr).T
            # Use the full dwell array to meet the requests.
            for i, request in enumerate(requests):
                prefix = request.property[:3]
                suffix = request.property[-4:]
                if suffix == 'dwel':
                    s = self.scene.state_space_shape
                    out = apply_reductions(s, request, full_dwell_array)
                    responses[i] = out.tolist()
        else:
            # Compute each reduction separately.
            for i, request in enumerate(requests):
                prefix = request.property[:3]
                suffix = request.property[-4:]
                if suffix != 'dwel':
                    continue

                # Compute the dwell object per process for the request.
                dwell_objects = []
                for p in self.scene.process_definitions:
                    obj = ImplicitDwellExpmFrechet(
                            self.scene.state_space_shape,
                            p.row_states,
                            p.column_states,
                            p.transition_rates,
                            request.state_reduction.states,
                            request.state_reduction.weights,
                            )
                    dwell_objects.append(obj)

                # Use the dwell object to compute the reduction.
                edge_to_dwell = expect.get_edge_to_site_expectations(
                        nsites, nstates,
                        self.expm_objects,
                        dwell_objects,
                        self.node_to_marginal_distn,
                        self.node_to_subtree_likelihoods,
                        self.prior_distn,
                        self.T,
                        self.root,
                        self.edges,
                        self.edge_rate_pairs,
                        self.edge_process_pairs,
                        self.scene.state_space_shape,
                        self.scene.observed_data.nodes,
                        self.scene.observed_data.variables,
                        self.scene.observed_data.iid_observations,
                        debug=False)

                # These dwell times will have been scaled
                # by the edge-specific scaling factor.
                # We want to remove that effect.
                for edge, edge_rate in self.edge_rate_pairs:
                    if edge_rate:
                        edge_to_dwell[edge] /= edge_rate

                # Map expectations back to edge indices.
                # The output will be like (nedges, nsites).
                edge_dwell_out = []
                for edge in self.edges:
                    dwell_expectations = edge_to_dwell[edge]
                    edge_dwell_out.append(dwell_expectations)

                # Transpose to (nsites, nedges).
                dwell_array = np.array(edge_dwell_out).T

                # Create a custom prefix indicating that the 'state' axis
                # reduction has already been performed.
                custom_prefix = list(request.property[:3])
                custom_prefix[2] = 'x'
                custom_prefix = ''.join(custom_prefix)

                # Compute the requested reduction using the custom prefix.
                out = apply_prefixed_reductions(
                    self.scene.state_space_shape,
                    custom_prefix,
                    request,
                    dwell_array)
                responses[i] = out.tolist()

        return True

    def _respond_to_tran(self, unmet_core_requests, requests, responses):
        if 'tran' not in unmet_core_requests:
            return False
        if self.node_to_subtree_likelihoods is None:
            return False
        if self.node_to_marginal_distn is None:
            return False
        for i, request in enumerate(requests):
            prefix = request.property[:3]
            suffix = request.property[-4:]
            if suffix != 'tran':
                continue

            # Create the request-specific expm transition objects.
            expm_transition_objects = []
            for p in self.scene.process_definitions:
                obj = ImplicitTransitionExpmFrechetEx(
                        self.scene.state_space_shape,
                        p.row_states,
                        p.column_states,
                        p.transition_rates,
                        request.transition_reduction.row_states,
                        request.transition_reduction.column_states,
                        request.transition_reduction.weights,
                        )
                expm_transition_objects.append(obj)
            arr = _compute_transition_expectations(
                self.scene,
                self.expm_objects,
                expm_transition_objects,
                self.node_to_marginal_distn,
                self.node_to_subtree_likelihoods,
                self.prior_distn,
                self.T,
                self.root,
                self.edges,
                self.edge_rate_pairs,
                self.edge_process_pairs,
                debug=False)
            out = np.array(arr).T

            # Apply further reductions.
            s = self.scene.state_space_shape
            out = apply_reductions(s, request, out)
            responses[i] = out.tolist()

        return True


    def react(self, requests, responses):
        """
        This is called repeatedly, with some progress made in each call.
        The two input lists should have equal lengths, and this function
        should eventually replace each None entry of the responses list
        with a useful entry after enough calls.

        """
        # Get the set of unmet core properties.
        unmet_core_requests = set()
        for request, response in zip(requests, responses):
            if response is None:
                unmet_core_requests.add(request.property[-4:])

        # Delete intermediate arrays.
        if self._delete_root_marginal_distn(unmet_core_requests):
            return self._note('delete root marginal distn')
        if self._delete_derivatives(unmet_core_requests):
            return self._note('delete derivatives')
        if self._delete_gradients(unmet_core_requests):
            return self._note('delete gradients')
        if self._delete_root_conditional_likelihoods(unmet_core_requests):
            return self._note('delete root conditional likelihoods')
        if self._delete_likelihoods(unmet_core_requests):
            return self._note('delete likelihoods')
        if self._delete_log_likelihoods(unmet_core_requests):
            return self._note('delete log likelihoods')
        if self._delete_node_to_conditional_likelihoods(unmet_core_requests):
            return self._note('delete node to conditional likelihoods')
        if self._delete_node_to_subtree_likelihoods(unmet_core_requests):
            return self._note('delete node to subtree likelihoods')
        if self._delete_node_to_marginal_distn(unmet_core_requests):
            return self._note('delete node to marginal distn')

        # Check feasibility.
        if self._check_feasibility():
            return self._note('check feasibility')

        # Respond to requests.
        if self._respond_to_root(unmet_core_requests, requests, responses):
            return self._note('respond to a "root" request')
        if self._respond_to_logl(unmet_core_requests, requests, responses):
            return self._note('respond to a "logl" request')
        if self._respond_to_deri(unmet_core_requests, requests, responses):
            return self._note('respond to a "deri" request')
        if self._respond_to_grad(unmet_core_requests, requests, responses):
            return self._note('respond to a "grad" request')
        if self._respond_to_node(unmet_core_requests, requests, responses):
            return self._note('respond to a "node" request')
        if self._respond_to_dwel(unmet_core_requests, requests, responses):
            return self._note('respond to a "dwel" request')
        if self._respond_to_tran(unmet_core_requests, requests, responses):
            return self._note('respond to a "tran" request')

        # Create intermediate arrays.
        if self._create_likelihoods(unmet_core_requests):
            return self._note('create likelihoods')
        if self._create_log_likelihoods(unmet_core_requests):
            return self._note('create log likelihoods')
        if self._create_node_to_conditional_likelihoods(unmet_core_requests):
            return self._note('create node to conditional likelihoods')
        if self._create_node_to_subtree_likelihoods(unmet_core_requests):
            return self._note('create node to subtree likelihoods')
        if self._create_derivatives(unmet_core_requests):
            return self._note('create derivatives')
        if self._create_gradients(unmet_core_requests):
            return self._note('create gradients')
        if self._create_root_conditional_likelihoods(unmet_core_requests):
            return self._note('create root conditional likelihoods')
        if self._create_root_marginal_distn(unmet_core_requests):
            return self._note('create root marginal distn')
        if self._create_node_to_marginal_distn(unmet_core_requests):
            return self._note('create node to marginal distn')

        # No progress was made towards an unfulfilled request.
        raise Exception(dir(self))


    def main(self, requests):
        responses = [None] * len(requests)
        try:
            while None in responses or not self.checked_feasibility:
                self.react(requests, responses)
            j_out = dict(
                    status = 'feasible',
                    responses = responses)
        except InfeasibilityError as e:
            j_out = dict(
                    status = 'infeasible',
                    responses = None)
        # require that feasibility has been checked
        assert_(self.checked_feasibility)
        return j_out


def process_json_in(j_in, debug=False):
    toplevel = TopLevel(j_in)
    return Reactor(toplevel.scene, debug=debug).main(toplevel.requests)
