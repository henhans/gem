###########################################
# Nambu (spin-down particle-hole) transformation for singlet superconductivity
# Author: Tsung-Han Lee
# Email:  henhans74716@gmail.com
###########################################
r"""
Superconducting (singlet BdG) calculations in GEM.

GEM solves particle-number conserving problems. A superconductor with singlet
(opposite-spin) pairing can be brought to that form with the particle-hole
transformation of the spin-down orbitals,

.. math::
    c_{k\uparrow} \to c_{k\uparrow}, \qquad
    c_{k\downarrow} \to d^\dagger_{-k\downarrow},

which maps the BdG Hamiltonian onto a normal Hamiltonian of the Nambu spinor
:math:`\psi_k = (c_{k\uparrow}, d_{k\downarrow})`. Pairing
:math:`\Delta_{ij} c^\dagger_{i\uparrow} c^\dagger_{j\downarrow}` becomes the
hybridization :math:`\Delta_{ij} c^\dagger_{i\uparrow} d_{j\downarrow}`, and the
conserved number of the transformed fermions is :math:`N_\uparrow - N_\downarrow`
plus a constant, i.e. the physical :math:`S_z`. Lattice, fragment and solvers
are then used unchanged on the transformed operators: the anomalous parts of
:math:`R`, :math:`\Lambda`, :math:`D` and :math:`\Lambda_c` are their
:math:`\uparrow d` blocks.

Conventions, the same as everywhere else in GEM:

* spin-orbitals are interleaved with spin as the fastest index; the even
  level ``2*a`` is called spin up and the odd level ``2*a+1`` spin down, as
  in the :math:`S_z = (n_{2a} - n_{2a+1})/2` of ``SimpleED``. It is the odd
  level that is particle-hole transformed: after the transformation index
  ``2*a+1`` is the hole :math:`d_a`. For singlet pairing the choice of the
  transformed spin is immaterial;
* one-body terms are :math:`\sum_{ij} h_{ij} c^\dagger_i c_j`;
* two-body terms are :math:`\frac12 \sum_{ijkl} U_{ijkl}
  c^\dagger_i c^\dagger_k c_l c_j` (as in ``SimpleED`` and
  :func:`gem.utilities.U_matrix_kanamori`).

The transformation of the chemical potential term
:math:`-\mu N = -\mu \sum_a \tau_3^{aa} n'_a + \text{const}` is handled by
:class:`gem.fragment.Fragment` with ``nambu=True``, see :func:`nambu_charge`.

Only singlet / opposite-spin pairing and :math:`S_z`-conserving normal terms
can be represented. Spin-flip one-body terms or equal-spin (triplet
:math:`S_z=\pm1`) pairing would break the conservation of the transformed
particle number and raise a ``ValueError``.
"""

import itertools as it
import numpy as np


def nambu_charge(n):
    r"""
    Charge :math:`\tau_3` of the transformed spin-orbitals: ``+1`` for the
    (particle) spin-up orbitals and ``-1`` for the (hole) spin-down orbitals.

    :param n: int. Number of spin-orbitals (even).

    Return:
        tau3: ndarray of shape (n,).
    """
    if n % 2 != 0:
        raise ValueError(f"the number of spin-orbitals must be even, got {n}")
    return np.tile([1.0, -1.0], n // 2)


def nambu_one_body(h, pairing=None, h_mk=None, tol=1e-12):
    r"""
    Transform a one-body Hamiltonian (and an optional singlet pairing field) to
    the Nambu basis.

    .. math::
        \sum_{ij} h_{ij} c^\dagger_i c_j
        + \sum_{ab} \left[ \Delta_{ab} c^\dagger_{a\uparrow} c^\dagger_{b\downarrow}
        + \text{h.c.} \right]
        = \sum_{ij} h'_{ij} \psi^\dagger_i \psi_j + E_0

    with :math:`h'_{\uparrow\uparrow} = h_{\uparrow\uparrow}`,
    :math:`h'_{dd} = -h^T_{\downarrow\downarrow}(-k)`,
    :math:`h'_{\uparrow d} = \Delta` and
    :math:`E_0 = \mathrm{Tr}\, h_{\downarrow\downarrow}`.

    Works on a single matrix ``(n,n)`` or on a stack ``(nk,n,n)`` (e.g. the
    dispersion :math:`\epsilon_k` handed to :class:`gem.lattice.Lattice`).

    :param h: ndarray (n,n) or (nk,n,n). One-body Hamiltonian in the
        spin-interleaved basis. Must conserve :math:`S_z`.
    :param pairing: ndarray (n/2,n/2) or (nk,n/2,n/2), optional. Pairing
        amplitudes :math:`\Delta_{ab}` of :math:`c^\dagger_{a\uparrow}
        c^\dagger_{b\downarrow}` (default: no pairing). Singlet pairing has
        :math:`\Delta = \Delta^T`.
    :param h_mk: ndarray, optional. :math:`h(-k)`, same shape as ``h``. Only
        needed for a k-resolved ``h`` without time-reversal symmetry. If None,
        a stack ``(nk,n,n)`` is taken to satisfy :math:`h(-k) = h(k)^*` (real
        hoppings), so that the hole block is simply
        :math:`-h_{\downarrow\downarrow}(k)`, and a single matrix ``(n,n)`` is
        taken to be local, so that the hole block is
        :math:`-h^T_{\downarrow\downarrow}`.
    :param tol: float, optional. Tolerance for the :math:`S_z` check.

    Return:
        h_nambu: ndarray, same shape as ``h``. One-body Hamiltonian of the
        transformed fermions.
        e0: float or ndarray (nk,). Constant :math:`\mathrm{Tr}\,
        h_{\downarrow\downarrow}`.
    """
    h = np.asarray(h)
    n = h.shape[-1]
    nambu_charge(n)
    if h.shape[-2] != n:
        raise ValueError(f"h must be square, got {h.shape}")
    if max(np.abs(h[..., ::2, 1::2]).max(), np.abs(h[..., 1::2, ::2]).max()) > tol:
        raise ValueError("h has spin-flip (up-down) elements: they cannot be "
                         "represented in the Nambu basis of singlet pairing")
    if h_mk is None:
        h_mk = h if h.ndim == 2 else h.conj()
    else:
        h_mk = np.asarray(h_mk)
        if h_mk.shape != h.shape:
            raise ValueError(f"h_mk must have the shape of h {h.shape}, got {h_mk.shape}")

    h_nambu = np.zeros(h.shape, dtype=np.result_type(h, h_mk, np.complex128))
    h_nambu[..., ::2, ::2] = h[..., ::2, ::2]
    h_nambu[..., 1::2, 1::2] = -np.swapaxes(h_mk[..., 1::2, 1::2], -1, -2)
    if pairing is not None:
        pairing = np.asarray(pairing)
        if pairing.shape[-2:] != (n // 2, n // 2):
            raise ValueError(f"pairing must be ({n//2},{n//2}), got {pairing.shape}")
        h_nambu[..., ::2, 1::2] += pairing
        h_nambu[..., 1::2, ::2] += np.swapaxes(pairing, -1, -2).conj()
    e0 = np.trace(h[..., 1::2, 1::2], axis1=-2, axis2=-1)
    # always complex: the k sums of Lattice are jitted for complex128
    return h_nambu, e0


def _normal_order(ops, coeff):
    """
    Normal order a product of ladder operators.

    :param ops: tuple of (index, dagger) pairs, read left to right.
    :param coeff: complex. Prefactor of the product.

    Return:
        dict mapping normal-ordered operator tuples (all creators on the left,
        original relative order kept) to their coefficient.
    """
    out = {}
    stack = [(tuple(ops), coeff)]
    while stack:
        ops, c = stack.pop()
        # first annihilator immediately followed by a creator
        pos = next((m for m in range(len(ops) - 1)
                    if not ops[m][1] and ops[m + 1][1]), None)
        if pos is None:
            out[ops] = out.get(ops, 0.0) + c
            continue
        a, b = ops[pos], ops[pos + 1]
        # a_x a^dag_y = delta_xy - a^dag_y a_x
        stack.append((ops[:pos] + (b, a) + ops[pos + 2:], -c))
        if a[0] == b[0]:
            stack.append((ops[:pos] + ops[pos + 2:], c))
    return out


def nambu_interaction(Utensor, tol=1e-12):
    r"""
    Transform a two-body interaction to the Nambu basis.

    .. math::
        \frac12 \sum_{ijkl} U_{ijkl} c^\dagger_i c^\dagger_k c_l c_j
        = \frac12 \sum_{ijkl} U'_{ijkl} \psi^\dagger_i \psi^\dagger_k \psi_l \psi_j
        + \sum_{ij} h'_{ij} \psi^\dagger_i \psi_j + E_0

    For the Hubbard interaction :math:`U n_\uparrow n_\downarrow` this gives
    :math:`-U n_\uparrow n_d + U n_\uparrow`.

    :param Utensor: ndarray (n,n,n,n). Interaction in the spin-interleaved
        basis. Must conserve :math:`S_z`.
    :param tol: float, optional. Elements below this are ignored.

    Return:
        U_nambu: ndarray (n,n,n,n). Two-body interaction of the transformed fermions.
        h_nambu: ndarray (n,n). One-body term generated by normal ordering.
        e0: float. Constant generated by normal ordering.
    """
    Utensor = np.asarray(Utensor)
    n = Utensor.shape[0]
    nambu_charge(n)
    if Utensor.shape != (n,) * 4:
        raise ValueError(f"Utensor must be ({n},{n},{n},{n}), got {Utensor.shape}")

    hole = lambda p: p % 2 == 1
    # c^dag_p -> d_p and c_p -> d^dag_p for the spin-down orbitals
    flip = lambda p, dag: (p, (not dag) if hole(p) else dag)

    U_nambu = np.zeros(Utensor.shape, dtype=np.complex128)
    h_nambu = np.zeros((n, n), dtype=np.complex128)
    e0 = 0.0 + 0.0j
    for i, j, k, l in it.product(range(n), repeat=4):
        v = Utensor[i, j, k, l]
        if abs(v) < tol or i == k or j == l:
            continue
        ops = (flip(i, True), flip(k, True), flip(l, False), flip(j, False))
        for term, c in _normal_order(ops, 0.5 * v).items():
            if abs(c) < tol:
                continue
            ndag = sum(d for _, d in term)
            if 2 * ndag != len(term):
                raise ValueError("Utensor does not conserve S_z: it cannot be "
                                 "represented in the Nambu basis of singlet pairing")
            if len(term) == 4:
                (p, _), (q, _), (r, _), (s, _) = term
                # psi^dag_p psi^dag_q psi_r psi_s = 1/2 U'[p,s,q,r] (...)
                U_nambu[p, s, q, r] += 2.0 * c
            elif len(term) == 2:
                h_nambu[term[0][0], term[1][0]] += c
            else:
                e0 += c

    if np.isrealobj(Utensor):
        U_nambu, h_nambu, e0 = U_nambu.real, h_nambu.real, e0.real
    return U_nambu, h_nambu, e0


def nambu_local(eloc, Utensor, pairing=None):
    r"""
    Local Hamiltonian of a fragment in the Nambu basis.

    Combines :func:`nambu_one_body` and :func:`nambu_interaction`: the returned
    ``eloc`` and ``Utensor`` are meant to be handed to a
    :class:`gem.fragment.Fragment` created with ``nambu=True``.

    :param eloc: ndarray (n,n). Local one-body Hamiltonian.
    :param Utensor: ndarray (n,n,n,n). Local interaction.
    :param pairing: ndarray (n/2,n/2), optional. Local singlet pairing field,
        see :func:`nambu_one_body`.

    Return:
        eloc_nambu: ndarray (n,n).
        U_nambu: ndarray (n,n,n,n).
        e0: float. Constant :math:`\langle H_{loc} \rangle - \langle H'_{loc} \rangle`.
    """
    h1, e1 = nambu_one_body(eloc, pairing=pairing)
    U_nambu, h2, e2 = nambu_interaction(Utensor)
    return h1 + h2, U_nambu, e1 + e2


def nambu_density_matrix(denMat):
    r"""
    Physical one-body density matrices from the density matrix of the
    transformed fermions, :math:`\rho'_{ij} = \langle \psi^\dagger_i \psi_j \rangle`.

    :param denMat: ndarray (n,n). Transformed density matrix, e.g.
        ``fragment.denMat[:nimp,:nimp]``.

    Return:
        rho_up: ndarray (n/2,n/2). :math:`\langle c^\dagger_{a\uparrow} c_{b\uparrow} \rangle`.
        rho_dn: ndarray (n/2,n/2). :math:`\langle c^\dagger_{a\downarrow} c_{b\downarrow} \rangle`.
        F: ndarray (n/2,n/2). Anomalous :math:`\langle c^\dagger_{a\uparrow} c^\dagger_{b\downarrow} \rangle`.
    """
    denMat = np.asarray(denMat)
    n = denMat.shape[0]
    nambu_charge(n)
    rho_up = denMat[::2, ::2]
    rho_dn = np.eye(n // 2) - denMat[1::2, 1::2].T
    F = denMat[::2, 1::2]
    return rho_up, rho_dn, F
