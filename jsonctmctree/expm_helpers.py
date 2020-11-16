"""
Implement various strategies to compute functions related to matrix exponential.

Some of these do precalculations, so they are not pure functions.

"""
from __future__ import division, print_function, absolute_import

import sys

import numpy as np
from numpy.testing import assert_equal, assert_

import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse import coo_matrix

from .pyexp import expm_multiply
from .pyexp.ctmc_ops import (
        Propagator, SmarterPropagator, ExplicitPropagator,
        MatrixExponential, RdOperator)
from .pyexp.linear_system import LinearSystem


__all__ = [
        'PadeExpm', 'EigenExpm', 'ActionExpm',
        'ExplicitExpmFrechet',
        'ImplicitDwellExpmFrechet',
        'ImplicitTransitionExpmFrechet',
        'ImplicitTransitionExpmFrechetEx',
        ]


def create_sparse_pre_rate_matrix(state_space_shape, row, col, rate):
    """
    Create the pre-rate matrix with empty diagonal.

    """
    # check conformability of input arrays
    ndim = len(state_space_shape)
    assert_equal(len(row.shape), 2)
    assert_equal(len(col.shape), 2)
    assert_equal(len(rate.shape), 1)
    assert_equal(row.shape[0], rate.shape[0])
    assert_equal(col.shape[0], rate.shape[0])
    assert_equal(row.shape[1], ndim)
    assert_equal(col.shape[1], ndim)

    # Define the transition rate matrix after the multivariate process
    # is collapsed to a univariate process.
    nstates = np.prod(state_space_shape)
    mrow = np.ravel_multi_index(row.T, state_space_shape)
    mcol = np.ravel_multi_index(col.T, state_space_shape)

    # Diagonal entries are allowed for computing dwell times,
    # but they are not allowed for transition expectations.
    #assert_(not np.any(mrow == mcol))

    # create the sparse pre_Q matrix from the sparse arrays
    return coo_matrix((rate, (mrow, mcol)), (nstates, nstates))


def create_sparse_rate_matrix(state_space_shape, row, col, rate):
    """
    Create the rate matrix.

    """
    nstates = np.prod(state_space_shape)
    Q = create_sparse_pre_rate_matrix(state_space_shape, row, col, rate)

    # get the dense array of exit rates, and set the diagonal
    exit_rates = Q.sum(axis=1).A.flatten()
    assert_equal(exit_rates.shape, (nstates, ))

    # FIXME
    # Temporarily disabling the setdiag of coo_matrix.
    # When it is OK to use new scipy, use the setdiag path instead.
    if False:
        Q.setdiag(-exit_rates)
    else:
        Q.row = np.concatenate((Q.row, np.arange(nstates)))
        Q.col = np.concatenate((Q.col, np.arange(nstates)))
        Q.data = np.concatenate((Q.data, -exit_rates))
        Q.has_canonical_format = False

    return Q


def create_dense_rate_matrix(state_space_shape, row, col, rate):
    """
    Create the rate matrix.

    """
    return create_sparse_rate_matrix(state_space_shape, row, col, rate).A


class PadeExpm(object):
    """
    This requires lower memory than EigenExpm.
    The implementation is the simplest because it does not cache anything.

    """
    def __init__(self, state_space_shape, row, col, rate):
        self.Q = create_sparse_rate_matrix(state_space_shape, row, col, rate)

    def expm_mul(self, rate_scaling_factor, A):
        """
        Compute exp(Q * r) * A.

        """
        return scipy.linalg.expm(self.Q.A * rate_scaling_factor).dot(A)

    def rate_mul(self, rate_scaling_factor, PA):
        """
        Compute Q * r * PA.
        This is for gradient calculation.

        """
        return rate_scaling_factor * self.Q.dot(PA)


#TODO more carefully treat matrices that can be permuted to block diagonal
#     with zeros on one of the diagonals.
class EigenExpm(object):
    def __init__(self, state_space_shape, row, col, rate):
        self.Q = create_dense_rate_matrix(state_space_shape, row, col, rate)
        self.w, self.U = scipy.linalg.eig(self.Q)
        self.V = scipy.linalg.inv(self.U)

    def expm_mul(self, rate_scaling_factor, A):
        """
        Compute exp(Q * r) * A.

        """
        w_exp = np.exp(self.w * rate_scaling_factor)
        VA = self.V.dot(A)
        return (self.U * w_exp).dot(VA).real

    def rate_mul(self, rate_scaling_factor, PA):
        """
        Compute Q * r * PA.
        This is for gradient calculation.

        """
        return rate_scaling_factor * self.Q.dot(PA)


class ActionExpm(object):
    """
    This uses a newer implementation of the matrix exponential vector product.

    Check some predicted costs and decide whether to use
    the abstract linear operator approach or the explicit matrix
    exponential approach.

    """
    def __init__(self, state_space_shape, row, col, rate, debug=False):
        R = create_sparse_pre_rate_matrix(state_space_shape, row, col, rate)
        self._L = LinearSystem(R)

    def expm_rmul(self, rate_scaling_factor, A):
        """
        Compute A * exp(Q * r).

        This uses the fact that exp(X.T) = exp(X).T.

        """
        P = MatrixExponential(self._L.propagator, rate_scaling_factor)
        return P.T.dot(A.T).T

    def expm_mul(self, rate_scaling_factor, A):
        """
        Compute exp(Q * r) * A.

        """
        P = MatrixExponential(self._L.propagator, rate_scaling_factor)
        return P.dot(A)

    def expm_tmul(self, rate_scaling_factor, A):
        """
        Compute exp(Q * r)' * A.

        """
        P = MatrixExponential(self._L.propagator, rate_scaling_factor)
        return P.T.dot(A)

    def rate_mul(self, rate_scaling_factor, PA):
        """
        Compute Q * r * PA.
        This is for gradient calculation.

        """
        Q = self._L.instantaneous_operator
        return rate_scaling_factor * Q.dot(PA)

    def gradient_red(self, rate_scaling_factor, left_vector, right_vector):
        """
        Compute r * (p' * Q' * q).
        This is for gradient calculation.

        """
        Q = self._L.instantaneous_operator
        return np.sum(Q.dot(left_vector) * right_vector, axis = 0) * rate_scaling_factor


class ActionExpmOld(object):
    def __init__(self, state_space_shape, row, col, rate, debug=False):
        self.Q = create_sparse_rate_matrix(state_space_shape, row, col, rate)
        self.debug = debug

    def _wrapped_expm_multiply(self, A, B):
        n = B.shape[0]
        if self.debug:
            (A.dot(np.identity(n))).tofile('A.txt')
            B.tofile('B.txt')
        return expm_multiply(A, B)

    def expm_rmul(self, rate_scaling_factor, A):
        """
        Compute A * exp(Q * r).

        This uses the fact that exp(X.T) = exp(X).T.

        """
        return self._wrapped_expm_multiply(
                rate_scaling_factor * self.Q.T, A.T).T

    def expm_mul(self, rate_scaling_factor, A):
        """
        Compute exp(Q * r) * A.

        """
        return self._wrapped_expm_multiply(
                rate_scaling_factor * self.Q, A)

    def rate_mul(self, rate_scaling_factor, PA):
        """
        Compute Q * r * PA.
        This is for gradient calculation.

        """
        return rate_scaling_factor * self.Q.dot(PA)


class ExplicitExpmFrechet(object):
    """
    This is for computing conditional expectations on edges.

    """
    def __init__(self, state_space_shape, row, col, rate, expect):
        self.Q = create_sparse_rate_matrix(
                state_space_shape, row, col, rate)
        self.E = create_sparse_pre_rate_matrix(
                state_space_shape, row, col, rate * expect)

    def get_expm_and_frechet(self, rate_scaling_factor):
        t = rate_scaling_factor
        QAt = (self.Q * t).A
        EAt = (self.E * t).A
        P, numerator = scipy.linalg.expm_frechet(QAt, EAt)
        del QAt
        del EAt
        return P, numerator


##############################################################################
# base class for implicit plain and extended dwell and transition expectations


class ImplicitExpmFrechetBase(object):
    def get_expm_frechet_product(self, rate_scaling_factor, A):
        """
        expm([[R - D, R o E],    A0
              [  0,   R - D]])   A1

        P A0 + K A1
        0 A0 + P A1

        Returns
        -------
        P dot A
        K dot A

        """
        AA = np.vstack((A, A))
        BB = expm_multiply(self.F * rate_scaling_factor, AA)
        PA = BB[self.nstates:]
        KA = BB[:self.nstates] - PA
        return PA, KA


#####################################################
# implicit expm frechet to compute dwell expectations


class ImplicitDwellExpmFrechet(ImplicitExpmFrechetBase):
    """
    For computing conditional dwell expectations on edges.

    """
    def __init__(self, state_space_shape, row, col, rate, s_state, s_weight):
        """
        Define a sparse rate matrix with shape (2n, 2n) where n is nstates.

        [[R - D,   E  ],
         [  0,   R - D]]

        This uses some folk theory of Frechet derivatives of matrix
        exponentiation that I picked up from Hobolth and Tataru
        and other miscellaneous sources, and it is combined with
        an algorithm by Al-Mohy et al.

        """
        nstates = np.prod(state_space_shape)
        self.nstates = nstates

        # Initialize the upper-left sparse matrix.
        Q00 = create_sparse_pre_rate_matrix(
                state_space_shape, row, col, rate)
        exit_rates = Q00.sum(axis=1).A.flatten()
        assert_equal(exit_rates.shape, (nstates, ))

        # Initialize the upper-right sparse diagonal matrix.
        Q01 = create_sparse_pre_rate_matrix(
                state_space_shape, s_state, s_state, s_weight)
        Q01.col += nstates

        # Define the diagonal entries of the upper-left sparse matrix.
        Q00.row = np.concatenate((Q00.row, np.arange(nstates)))
        Q00.col = np.concatenate((Q00.col, np.arange(nstates)))
        Q00.data = np.concatenate((Q00.data, -exit_rates))
        Q00.has_canonical_format = False

        # Initialize the lower-right sparse matrix.
        Q11 = Q00.copy()
        Q11.row += nstates
        Q11.col += nstates

        # Define the full matrix to be exponentiated.
        F_row = np.concatenate([Q00.row, Q01.row, Q11.row])
        F_col = np.concatenate([Q00.col, Q01.col, Q11.col])
        F_data = np.concatenate([Q00.data, Q01.data, Q11.data])
        F = coo_matrix((F_data, (F_row, F_col)), (2*nstates, 2*nstates))

        # Store the full matrix.
        self.F = F


##########################################################
# implicit expm frechet to compute transition expectations


class ImplicitTransitionExpmFrechet(ImplicitExpmFrechetBase):
    """
    For computing conditional transition count expectations on edges.

    Experimentally try using an expm-frechet-vector-product.

    If Jij is the 2d joint distribution over states at the endpoints of an edge,
    then I want to compute the following to get the expectation.
    sum_ij ( Jij * expm_frechet(Q, Q o E)ij / expm(Q)ij )
    The trick is that it may be possible to compute this
    using only matrix-vector products.
    
    This is possible.  See the tests directory for more details.

    """
    def __init__(self, state_space_shape, row, col, rate, expect):
        """
        Define a sparse rate matrix with shape (2n, 2n) where n is nstates.

        [[R - D, R o E],
         [  0,   R - D]]

        This uses some folk theory of Frechet derivatives of matrix
        exponentiation that I picked up from Hobolth and Tataru
        and other miscellaneous sources, and it is combined with
        an algorithm by Al-Mohy et al.

        """
        nstates = np.prod(state_space_shape)
        self.nstates = nstates

        # Initialize the upper-left sparse matrix.
        Q00 = create_sparse_pre_rate_matrix(
                state_space_shape, row, col, rate)
        exit_rates = Q00.sum(axis=1).A.flatten()
        assert_equal(exit_rates.shape, (nstates, ))

        # Initialize the upper-right sparse matrix.
        Q01 = create_sparse_pre_rate_matrix(
                state_space_shape, row, col, rate * expect)
        Q01.col += nstates

        # Define the diagonal entries of the upper-left sparse matrix.
        Q00.row = np.concatenate((Q00.row, np.arange(nstates)))
        Q00.col = np.concatenate((Q00.col, np.arange(nstates)))
        Q00.data = np.concatenate((Q00.data, -exit_rates))
        Q00.has_canonical_format = False

        # Initialize the lower-right sparse matrix.
        Q11 = Q00.copy()
        Q11.row += nstates
        Q11.col += nstates

        # Define the full matrix to be exponentiated.
        F_row = np.concatenate([Q00.row, Q01.row, Q11.row])
        F_col = np.concatenate([Q00.col, Q01.col, Q11.col])
        F_data = np.concatenate([Q00.data, Q01.data, Q11.data])
        F = coo_matrix((F_data, (F_row, F_col)), (2*nstates, 2*nstates))

        # Store the full matrix.
        self.F = F


class ImplicitTransitionExpmFrechetEx(ImplicitExpmFrechetBase):
    """
    Allow more flexibility by not equating the sparsity structures of R and E.

    """
    def __init__(self, state_space_shape,
            row, col, rate,
            expect_row, expect_col, expect_rate):
        """
        Define a sparse rate matrix with shape (2n, 2n) where n is nstates.

        [[R - D, R o E],
         [  0,   R - D]]

        Parameters
        ----------
        row : 2d integer array
            a sequence of multivariate row states
        col : 2d integer array
            a sequence of multivariate col states
        rate : 1d float array
            a sequence of floating point rates
        expect_row : 2d integer array
            a sequence of multivariate row states
        expect_col : 2d integer array
            a sequence of multivariate col states
        expect_rate : 1d float array
            a sequence of floating point rates

        """
        nstates = np.prod(state_space_shape)
        self.nstates = nstates

        # Initialize the upper-left sparse matrix.
        Q00 = create_sparse_pre_rate_matrix(
                state_space_shape, row, col, rate)
        exit_rates = Q00.sum(axis=1).A.flatten()
        assert_equal(exit_rates.shape, (nstates, ))

        # Initialize the upper-right sparse matrix.
        # Use sparse matrix elementwise multiplication.
        R = create_sparse_pre_rate_matrix(
                state_space_shape, expect_row, expect_col, expect_rate)
        Q01 = Q00.multiply(R).tocoo()
        Q01.col += nstates
        Q01.has_canonical_format = False

        # Define the diagonal entries of the upper-left sparse matrix.
        Q00.row = np.concatenate((Q00.row, np.arange(nstates)))
        Q00.col = np.concatenate((Q00.col, np.arange(nstates)))
        Q00.data = np.concatenate((Q00.data, -exit_rates))
        Q00.has_canonical_format = False

        # Initialize the lower-right sparse matrix.
        Q11 = Q00.copy()
        Q11.row += nstates
        Q11.col += nstates

        # Define the full matrix to be exponentiated.
        F_row = np.concatenate([Q00.row, Q01.row, Q11.row])
        F_col = np.concatenate([Q00.col, Q01.col, Q11.col])
        F_data = np.concatenate([Q00.data, Q01.data, Q11.data])
        F = coo_matrix((F_data, (F_row, F_col)), (2*nstates, 2*nstates))

        # Store the full matrix.
        self.F = F
