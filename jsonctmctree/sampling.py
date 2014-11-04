"""
This is for testing only.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg


def assert_symmetric_matrix(M):
    assert_allclose(M, M.T)

def sample_symmetric_rates(n):
    X = np.random.randn(n, n)
    S = np.exp(X + X.T)
    np.fill_diagonal(S, 0)
    return S

def sample_distn(n):
    d = np.exp(np.random.randn(n))
    return d / d.sum()

def sample_time_reversible_rate_matrix(nstates):
    d = sample_distn(nstates)
    Q = sample_symmetric_rates(nstates) * d
    Q -= np.diag(Q.sum(axis=1))
    assert_allclose(Q.sum(axis=1), 0, atol=1e-13)
    assert_allclose(d.dot(Q), 0, atol=1e-13)
    P = np.diag(d)
    assert_symmetric_matrix(P.dot(Q))
    return Q, d

def sample_time_nonreversible_rate_matrix(nstates):
    Q = np.exp(np.random.randn(nstates, nstates))
    Q -= np.diag(Q.sum(axis=1))
    w, V = scipy.linalg.eig(Q, left=True, right=False)
    i = np.argmin(np.abs(w))
    d = V[:, i].real
    d /= d.sum()
    assert_allclose(Q.sum(axis=1), 0, atol=1e-13)
    assert_allclose(d.dot(Q), 0, atol=1e-13)
    P = np.diag(d)
    return Q, d