"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_equal, assert_array_less, assert_allclose

import scipy.sparse
from scipy.linalg import expm

import jsonctmctree
from jsonctmctree.pyexp.basic_ops import PowerOperator, ExtendedMatrixOperator
from jsonctmctree.pyexp.ctmc_ops import (
        RdOperator, RdcOperator, RdCOperator,
        Propagator, SmarterPropagator, MatrixExponential)


def get_random_rate_matrix(n):
    """
    Returns a scipy sparse rate matrix without entries on the diagonal.
    """
    row_ind = []
    col_ind = []
    shape = (n, n)
    mask = np.random.randint(0, 2, size=shape)
    for i in range(n):
        for j in range(n):
            if i != j and mask[i, j]:
                row_ind.append(i)
                col_ind.append(j)
    ndata = len(row_ind)
    data = np.exp(np.random.randn(ndata))
    R = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=shape)
    return R


def get_random_sparse_square_matrix(n):
    """
    Returns a scipy sparse matrix.
    The matrix is square with floating-point type.
    """
    A = scipy.sparse.construct.rand(n, n, density=0.2)
    B = scipy.sparse.construct.rand(n, n, density=0.2)
    R = (A - B).tocsr()
    return R


def check_operator_equivalence(L, M):
    # Check properties of the operator, its transpose, and its adjoint.
    # This is restricted to real square operators.
    n = L.shape[1]
    B = np.random.randn(n, 2)
    for f, g in (L, M), (L.T, M.T), (L.H, M.T):
        # Check the action of the operator.
        assert_allclose(f.dot(B), g.dot(B))
        # Check the 1-norm and the infinity-norm.
        assert_allclose(f.one_norm(), np.linalg.norm(g, 1))
        assert_allclose(f.inf_norm(), np.linalg.norm(g, np.inf))


def test_RdOperator():
    # This is an n x n square operator.
    np.random.seed(1234)
    n = 4
    R = get_random_rate_matrix(n)
    d = np.random.randn(n)

    # Define the dense numpy ndarray.
    M = R.A + np.diag(d)

    # Define the linear operator.
    L = RdOperator(R, d)

    # Test properties of the operator, its transpose, and its adjoint.
    check_operator_equivalence(L, M)


def test_RdcOperator():
    # This is a 2n x 2n square operator.
    np.random.seed(1234)
    n = 4
    R = get_random_rate_matrix(n)
    d = np.random.randn(n)
    c = np.random.randn(n)

    # Define the dense numpy ndarray.
    Q = R.A + np.diag(d)
    M = np.bmat([[Q, np.diag(c)], [np.zeros((n, n)), Q]]).A

    # Define the linear operator.
    L_Rd = RdOperator(R, d)
    L = RdcOperator(L_Rd, c)

    # Test properties of the operator, its transpose, and its adjoint.
    check_operator_equivalence(L, M)


def test_RdCOperator():
    # This is a 2n x 2n square operator.
    np.random.seed(1234)
    n = 4
    R = get_random_rate_matrix(n)
    d = np.random.randn(n)
    C = get_random_sparse_square_matrix(n)

    # Define the dense numpy ndarray.
    Q = R.A + np.diag(d)
    M = np.bmat([[Q, C.A], [np.zeros((n, n)), Q]]).A

    # Define the linear operator.
    L_Rd = RdOperator(R, d)
    L = RdCOperator(L_Rd, C)

    # Test properties of the operator, its transpose, and its adjoint.
    check_operator_equivalence(L, M)


def test_PowerOperator():
    # This is an n x 2 square operator.
    np.random.seed(1234)
    n = 4
    C = get_random_sparse_square_matrix(n)
    B = np.random.randn(n, 2)
    for p in range(5):

        # Define the dense numpy ndarray.
        M = np.linalg.matrix_power(C.A, p)

        # Define the linear operator.
        L = PowerOperator(ExtendedMatrixOperator(C), p)

        # Test properties of the operator, its transpose, and its adjoint.
        for f, g in (L, M), (L.T, M.T), (L.H, M.T):
            assert_allclose(f.dot(B), g.dot(B))


def _sample_Rd(n):
    R = get_random_rate_matrix(n)
    d = np.random.randn(n)
    return RdOperator(R, d)


def _sample_Rdc(n):
    Rd = _sample_Rd(n)
    c = np.random.randn(n)
    return RdcOperator(Rd, c)


def _sample_RdC(n):
    Rd = _sample_Rd(n)
    C = get_random_sparse_square_matrix(n)
    return RdCOperator(Rd, C)


def test_MatrixExponential_zero_mu():
    # This is an n x n square operator.
    np.random.seed(1234)
    n = 4
    mu = 0
    for t in 0.42, 4.2, 42.0:
        for op in _sample_Rd(n), _sample_Rdc(n), _sample_RdC(n):
            k = op.shape[1]
            I = np.identity(k)
            P = Propagator(op, mu)
            L = MatrixExponential(P, t)
            for n0 in 2, 10:
                B = np.random.randn(k, n0)
                actual = L.dot(B)
                desired = expm(op.dot(I) * t).dot(B)
                assert_allclose(actual, desired)


def test_MatrixExponential_nontrivial_mu():
    # This is an n x n square operator.
    np.random.seed(1234)
    n = 4

    # Sample the operator with nontrivial mu to reduce the norm.
    R = get_random_rate_matrix(n)
    d = np.random.randn(n)
    mu = np.mean(d)
    op = RdOperator(R, d - mu)
    P = Propagator(op, mu)
    k = op.shape[1]

    # Compare to the brute force calculation across several scaling factors.
    for t in 0.42, 4.2, 42.0:
        L = MatrixExponential(P, t)
        for n0 in 2, 10:
            B = np.random.randn(k, n0)
            desired = expm((R.A + np.diag(d)) * t).dot(B)
            actual = L.dot(B)
            assert_allclose(actual, desired)


def test_SmarterPropagator():
    # This is an n x n square operator.
    np.random.seed(1234)
    n = 4

    # Sample a sparse matrix of rates.
    R = get_random_rate_matrix(n)

    # Create the 'smart' propagator.
    P = SmarterPropagator(R)

    # Create the rate matrix including explicit negative diagonal entries.
    exit_rates = R.sum(axis=1).A.ravel()
    Q = R - np.diag(exit_rates)

    # Compare to the brute force calculation across several scaling factors.
    for t in 0.42, 4.2, 42.0:
        L = MatrixExponential(P, t)
        for n0 in 2, 10:
            B = np.random.randn(n, n0)
            desired = expm(Q * t).dot(B)
            actual = L.dot(B)
            assert_allclose(actual, desired)
