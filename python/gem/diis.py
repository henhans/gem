###########################################
# DIIS (Pulay) extrapolation for the GEM self-consistency loop
# Adapted from the DIIS class of the earlier triqs_ghostGA code, itself taken from
#   PyDMET: a python implementation of density matrix embedding theory
#   Copyright (C) 2014, 2015 Sebastian Wouters
#   (GNU General Public License, version 2 or any later version)
###########################################
r"""
Direct inversion in the iterative subspace (DIIS, Pulay mixing).

For a fixed-point problem :math:`x = G(x)` the error of an iterate is
:math:`e_i = G(x_i) - x_i`. DIIS keeps the last few pairs
:math:`(e_i, G(x_i))` and returns the combination
:math:`\sum_i c_i G(x_i)` whose error :math:`\sum_i c_i e_i` has the smallest
norm, with :math:`\sum_i c_i = 1`. Unlike linear mixing it also converges to
fixed points that are unstable under the plain iteration, which is the case
of the paramagnetic superconducting solutions of :ref:`E4 <example_E4>`.

The states are combined linearly, so they must all be expressed in the same
gauge: in GEM, the bath gauge :math:`R \to U R`, :math:`\Lambda \to U \Lambda
U^\dagger` has to be fixed before the parameters are appended (e.g. by taking
:math:`R` Hermitian for ``B=1``). The coefficients are real, so that a
combination of Hermitian matrices stays Hermitian.
"""

import numpy as np


class DIIS:
    '''
    DIIS extrapolation over the last ``nvec`` iterates.

    Typical use in a self-consistency loop::

        diis = DIIS(8)
        for it in range(itmax):
            x_new = G(x)
            diis.append(x_new - x, x_new)
            x = diis.solve()
    '''

    def __init__(self, nvec=8):
        '''
        :param nvec: int, optional. Number of iterates kept (default: 8).
        '''
        if not isinstance(nvec, int) or nvec < 1:
            raise ValueError(f"nvec must be a positive int, got {nvec}")
        self.nvec = nvec
        self.errors = []
        self.states = []

    def append(self, error, state):
        '''
        Add an iterate, dropping the oldest one beyond ``nvec``.

        :param error: ndarray. Error of the iterate, :math:`G(x_i) - x_i`.
        :param state: ndarray. The new state :math:`G(x_i)`, any shape.
        '''
        self.errors.append(np.asarray(error).ravel().copy())
        self.states.append(np.asarray(state).copy())
        if len(self.errors) > self.nvec:
            self.errors.pop(0)
            self.states.pop(0)

    def flush(self):
        '''Forget the stored iterates, e.g. after a change of parameters.'''
        self.errors = []
        self.states = []

    def coefficients(self):
        '''
        DIIS coefficients, minimising :math:`|\\sum_i c_i e_i|` with :math:`\\sum_i c_i = 1`.

        Return:
            c: ndarray of real coefficients, oldest iterate first.
        '''
        n = len(self.errors)
        if n == 0:
            raise ValueError("DIIS has no stored iterates")
        B = np.zeros((n + 1, n + 1))
        for i in range(n):
            for j in range(i, n):
                B[i, j] = B[j, i] = np.real(np.vdot(self.errors[i], self.errors[j]))
        B[:n, n] = B[n, :n] = 1.0
        rhs = np.zeros(n + 1)
        rhs[n] = 1.0
        # least squares: the overlap matrix becomes singular when iterates repeat
        return np.linalg.lstsq(B, rhs, rcond=None)[0][:n]

    def solve(self):
        '''
        Extrapolated state :math:`\\sum_i c_i G(x_i)`.

        Return:
            ndarray with the shape of the stored states.
        '''
        c = self.coefficients()
        return sum(ci * si for ci, si in zip(c, self.states))
