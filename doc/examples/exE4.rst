.. _example_E4:

E4 — d-wave Superconductivity in the 2x2 Cluster Hubbard Model
==============================================================

**Scripts:** ``examples/E4/Square_2x2_dwave_sc.py``

This example computes the d-wave superconducting order parameter of the doped
Hubbard model on the square lattice at :math:`U = 8|t|`, with a :math:`2\times2`
cluster fragment at :math:`B = 1` (cluster RISB), as a function of the filling
:math:`n` (electrons per cluster). It uses the Nambu basis of
:mod:`gem.nambu` described in :ref:`superconductivity`.

Setup
-----

The four sites with two spins give ``nimp = 8`` and ``ntot = 16``. The cluster
dispersion :math:`\epsilon_{ij}(K) = \sum_R t_{ij}(R) e^{iK\cdot R}` is built on
an ``nk x nk`` grid of the supercell Brillouin zone; its :math:`K` average, the
intra-cluster hopping, is moved to ``eloc``. Lattice and local Hamiltonian are
then transformed to the Nambu basis, the latter with a small d-wave pairing
field :math:`\Delta_{ij} = \pm h/\sqrt{2}` on the :math:`x` (:math:`+`) and
:math:`y` (:math:`-`) bonds, which selects the symmetry of the order parameter::

    eks_n, _ = nambu_one_body(eks)
    lattice = Lattice(eks_n)
    eloc_n, U_n, e0 = nambu_local(eloc, Utensor, pairing=field/np.sqrt(2)*dwave)

    edsolver = SimpleED(ntot, use_Ntot=True, N_sector=ntot//2)
    fragment = Fragment(nimp, nbath, eloc_n, U_n, edsolver, nambu=True)

The transformed particle number is the physical :math:`S_z`, so the solver
conserves ``Ntot`` only, and the half-filled sector is the physical
:math:`S_z = 0`.

Self-consistency loop
---------------------

The loop is the usual one, with the filling fixed by
``fit_mu(mode='fragment')`` (the quasiparticle mode counts the transformed
particles, i.e. :math:`S_z`, and is refused for a Nambu fragment). During the
first ``nseed`` iterations a larger seed field drives the system to the
superconducting solution::

    for it in range(itmax):
        lattice.solve_qp([fragment], T=0.0, Tsmearing=Tsmearing)
        fragment.update_hybridization(T=0.0)
        fragment.impose_nambu_spin_symmetry()

        fragment.eloc = eloc_seed if (i_n == 0 and it < nseed) else eloc_n
        mu = lattice.fit_mu(n_target, [fragment], T=0.0, mu_old=mu,
                            mode='fragment', ntol=1e-8)

        Lambda_old, R_old = fragment.Lambda.copy(), fragment.R.copy()
        fragment.update_self_energy(T=0.0)
        fragment.impose_nambu_spin_symmetry()

        fragment.Lambda = (1 - mix)*fragment.Lambda + mix*Lambda_old
        fragment.R      = (1 - mix)*fragment.R + mix*R_old

:meth:`~gem.fragment.Fragment.impose_nambu_spin_symmetry` is the Nambu
counterpart of :meth:`~gem.fragment.Fragment.impose_spin_SU2_symmetry`: it
projects the parameters onto those invariant under a :math:`\pi` spin rotation,
which removes magnetization and keeps singlet pairing. It is switched off with
``paramagnetic = False``. Near half filling the paramagnetic d-wave solution is
then unstable, and the loop flows to Néel antiferromagnetism coexisting with
d-wave superconductivity, of lower energy (at :math:`n = 3.6`,
:math:`E_{tot} = -2.658` instead of :math:`-2.539`, staggered moment
:math:`0.52`).

Observables
-----------

The anomalous amplitudes
:math:`F_{ij} = \langle c^\dagger_{i\uparrow} c^\dagger_{j\downarrow} \rangle`
are given by :func:`~gem.nambu.nambu_density_matrix`, and the d-wave order
parameter is :math:`\sum_{\langle ij \rangle} \pm F_{ij} / (4\sqrt{2})`.
Since :math:`-\mu N` is :math:`-\mu \sum_a \tau_3^{aa} n'_a` plus a constant
and the interaction has been normal ordered, the physical local energy is::

    eloc_phys = fragment.compute_energy() + mu*np.dot(fragment.charge, np.diag(dm)) + e0

The script stores :math:`\mu`, :math:`Z`, the order parameter, the kinetic,
potential and total energy for every filling in
``data_Square_2x2_dwave_U8.0.h5``.

Results
-------

The d-wave order parameter has a dome as a function of filling, with its
maximum near :math:`n = 3.7` and a remnant at half filling that is induced by
the field :math:`h = 10^{-3}`. All quantities agree, to the convergence
tolerance, with an independent full-Nambu cluster-RISB implementation
(``simple_ga``):

.. list-table::
   :header-rows: 1

   * - :math:`n`
     - :math:`\mu`
     - :math:`Z`
     - d-wave
     - :math:`E_{tot}`
   * - 3.6
     - 2.2486
     - 0.6088
     - 0.0786
     - -2.5391
   * - 3.7
     - 2.3513
     - 0.6043
     - 0.0799
     - -2.3091
   * - 3.8
     - 2.4534
     - 0.6141
     - 0.0756
     - -2.0689
   * - 3.9
     - 2.5901
     - 0.6400
     - 0.0626
     - -1.8176
   * - 4.0
     - 4.0000
     - 0.6587
     - 0.0329
     - -1.5229

Details of the comparison are given in ``examples/E4/README.md``.
