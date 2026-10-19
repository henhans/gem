'''Nambu (spin-down particle-hole) transformation for singlet superconductivity.

* The transformed local Hamiltonian has the spectrum of the physical one in the
  full Fock space, for Kanamori interaction, Sz-conserving hopping and a
  singlet pairing field (brute force, Jordan-Wigner matrices).
* Terms that break Sz are refused.
* A non-interacting BCS lattice with a local s-wave field: RISB (B=1) is exact
  there, so the self-consistent fragment must reproduce the BdG k-sum, filling
  fitted with ``fit_mu(mode='fragment')``.
'''

import unittest
import itertools as it

import numpy as np

from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
from gem.utilities import U_matrix_kanamori, calc_nf
from gem.nambu import (nambu_charge, nambu_one_body, nambu_interaction,
                       nambu_local, nambu_density_matrix)


def fock_annihilators(n):
    '''Jordan-Wigner c_p as dense (2^n,2^n) matrices.'''
    a = np.array([[0., 1.], [0., 0.]])
    z = np.diag([1., -1.])
    ops = []
    for p in range(n):
        m = np.array([[1.]])
        for q in range(n):
            m = np.kron(m, z if q < p else (a if q == p else np.eye(2)))
        ops.append(m)
    return ops


def fock_hamiltonian(h1, U, c, pairing=None):
    '''sum h_ij c^dag_i c_j + 1/2 sum U_ijkl c^dag_i c^dag_k c_l c_j (+ pairing).'''
    n = h1.shape[0]
    cd = [x.T.conj() for x in c]
    H = np.zeros(c[0].shape, dtype=complex)
    for i, j in it.product(range(n), repeat=2):
        H += h1[i, j] * cd[i] @ c[j]
    for i, j, k, l in it.product(range(n), repeat=4):
        if U[i, j, k, l] != 0:
            H += 0.5 * U[i, j, k, l] * cd[i] @ cd[k] @ c[l] @ c[j]
    if pairing is not None:
        for a, b in it.product(range(n // 2), repeat=2):
            P = pairing[a, b] * cd[2 * a] @ cd[2 * b + 1]
            H += P + P.T.conj()
    return H


class test_nambu(unittest.TestCase):

    def test_charge(self):
        np.testing.assert_array_equal(nambu_charge(4), [1, -1, 1, -1])
        with self.assertRaises(ValueError):
            nambu_charge(3)

    def test_spectrum_kanamori(self):
        norb, mu = 2, 0.7
        n = 2 * norb
        rng = np.random.default_rng(7)
        t = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
        t = t + t.T.conj()
        h1 = np.kron(t, np.eye(2)) - mu * np.eye(n)
        U = U_matrix_kanamori(norb, 3.0, 0.6)
        pairing = rng.normal(size=(norb, norb)) + 1j * rng.normal(size=(norb, norb))
        pairing = pairing + pairing.T            # singlet

        c = fock_annihilators(n)
        E_phys = np.linalg.eigvalsh(fock_hamiltonian(h1, U, c, pairing))

        eloc_n, U_n, e0 = nambu_local(h1, U, pairing=pairing)
        E_nambu = np.linalg.eigvalsh(fock_hamiltonian(eloc_n, U_n, c)) + e0

        np.testing.assert_allclose(E_nambu, E_phys, atol=1e-10)

    def test_hubbard(self):
        U = np.zeros((2,) * 4)
        U[0, 0, 1, 1] = U[1, 1, 0, 0] = 4.0
        U_n, h_n, e0 = nambu_interaction(U)
        # U n_up n_dn = -U n_up n_d + U n_up
        np.testing.assert_allclose(U_n[0, 0, 1, 1] + U_n[1, 1, 0, 0], -8.0)
        np.testing.assert_allclose(h_n, np.diag([4.0, 0.0]))
        self.assertAlmostEqual(e0, 0.0)

    def test_dispersion_hole_block(self):
        rng = np.random.default_rng(3)
        ek = rng.normal(size=(5, 2, 2)) + 1j * rng.normal(size=(5, 2, 2))
        ek = np.kron(ek + ek.conj().transpose(0, 2, 1), np.eye(2))
        eks_n, e0 = nambu_one_body(ek)
        np.testing.assert_allclose(eks_n[:, 1::2, 1::2], -ek[:, 1::2, 1::2])
        np.testing.assert_allclose(e0, np.trace(ek[:, 1::2, 1::2], axis1=1, axis2=2))

    def test_refuse_sz_breaking(self):
        h = np.zeros((2, 2)); h[0, 1] = h[1, 0] = 1.0
        with self.assertRaises(ValueError):
            nambu_one_body(h)
        # c^dag_{0 up} c^dag_{1 up} c_{1 up} c_{0 dn}: moves a spin down to up
        U = np.zeros((4,) * 4); U[0, 1, 2, 2] = 1.0
        with self.assertRaises(ValueError):
            nambu_interaction(U)

    def test_fragment_options(self):
        nimp, nbath = 2, 2
        frag = Fragment(nimp, nbath, np.zeros((2, 2)), np.zeros((2,) * 4),
                        SimpleED(4, use_Ntot=True), nambu=True)
        np.testing.assert_array_equal(frag.charge, [1, -1])
        with self.assertRaises(ValueError):
            frag.impose_spin_SU2_symmetry()
        with self.assertRaises(NotImplementedError):
            Lattice(np.zeros((1, 2, 2))).fit_mu(1.0, [frag], T=0.0, Tsmearing=1e-2)

    def test_bcs_noninteracting(self):
        nimp, B = 2, 1
        nbath = nimp * B
        ntot = nimp + nbath
        t, delta0, n_target, T = 0.5, 0.2, 0.8, 0.02

        nk = 40
        kx, ky = np.meshgrid(*(2 * [np.linspace(-np.pi, np.pi, nk, endpoint=False)]))
        ek = (-2 * t * (np.cos(kx) + np.cos(ky))).ravel()
        eks = np.kron(ek[:, None, None], np.eye(2))
        eks_n, _ = nambu_one_body(eks)
        lattice = Lattice(eks_n)

        eloc_n, U_n, _ = nambu_local(np.zeros((2, 2)), np.zeros((2,) * 4),
                                    pairing=np.array([[delta0]]))
        solver = SimpleED(ntot, use_Ntot=True, N_sector=ntot // 2)
        frag = Fragment(nimp, nbath, eloc_n, U_n, solver, nambu=True)

        mu = 0.0
        for _ in range(200):
            lattice.solve_qp([frag], T=T)
            frag.update_hybridization(T=T)
            mu = lattice.fit_mu(n_target, [frag], T=T, mu_old=mu, mode='fragment', ntol=1e-8)
            L_old, R_old = frag.Lambda.copy(), frag.R.copy()
            frag.update_self_energy(T=T)
            if max(abs(frag.Lambda - L_old).max(), abs(frag.R - R_old).max()) < 1e-8:
                break

        # exact BdG at the same mu
        dm = np.zeros((2, 2), dtype=complex)
        hloc = eloc_n - mu * np.diag(nambu_charge(2))
        for e in eks_n:
            dm += calc_nf(e + hloc, T).T / len(eks_n)
        _, _, F_exact = nambu_density_matrix(dm)
        n_exact = np.real(dm[0, 0] + 1 - dm[1, 1])

        rho_up, rho_dn, F = nambu_density_matrix(frag.denMat[:nimp, :nimp])
        self.assertAlmostEqual(float(np.real(frag.nfill)), n_target, places=6)
        self.assertAlmostEqual(n_exact, n_target, places=5)
        np.testing.assert_allclose(rho_up, rho_dn, atol=1e-6)
        np.testing.assert_allclose(F, F_exact, atol=1e-5)
        self.assertGreater(abs(F[0, 0]), 1e-2)

        # the paramagnetic solution is invariant under the spin projector
        params = [frag.R.copy(), frag.Lambda.copy(), frag.D.copy(), frag.Lambda_c.copy()]
        frag.impose_nambu_spin_symmetry()
        for old, new in zip(params, [frag.R, frag.Lambda, frag.D, frag.Lambda_c]):
            np.testing.assert_allclose(new, old, atol=1e-6)

    def test_nambu_spin_projector(self):
        nimp, nbath = 4, 8
        frag = Fragment(nimp, nbath, np.zeros((4, 4)), np.zeros((4,) * 4),
                        SimpleED(12, use_Ntot=True), nambu=True)
        rng = np.random.default_rng(5)
        M = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
        frag.Lambda = M + M.conj().T
        frag.R = rng.normal(size=(8, 4)) + 1j * rng.normal(size=(8, 4))
        frag.impose_nambu_spin_symmetry()
        R, L = frag.R.copy(), frag.Lambda.copy()
        frag.impose_nambu_spin_symmetry()          # a projector: idempotent
        np.testing.assert_allclose(frag.R, R, atol=1e-14)
        np.testing.assert_allclose(frag.Lambda, L, atol=1e-14)
        np.testing.assert_allclose(L, L.conj().T, atol=1e-14)
        # no magnetization term n_up - n_dn = n_up + n_d - 1 in Lambda
        self.assertAlmostEqual(np.trace(L).real, 0.0, places=12)


if __name__ == '__main__':
    unittest.main()
