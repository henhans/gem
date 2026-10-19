.. _example_E4:

E4 — d-wave Superconductivity in the 2x2 Cluster Hubbard Model
==============================================================

**Scripts:** ``examples/E4/Square_2x2_dwave_sc.py``, ``examples/E4/plot_dwave.py``

This example computes the d-wave superconducting order parameter of the doped
Hubbard model on the square lattice at :math:`U = 8|t|`, with a :math:`2\times2`
cluster fragment at :math:`B = 1` (cluster RISB), as a function of the filling
:math:`n` (electrons per cluster). It uses the Nambu basis of
:mod:`gem.nambu` described in :ref:`superconductivity`, the non-magnetic
projection :meth:`~gem.fragment.Fragment.impose_nambu_spin_symmetry` and
DIIS extrapolation from :mod:`gem.diis`.

Setup
-----

The four sites with two spins give ``nimp = 8`` and ``ntot = 16``. The cluster
dispersion :math:`\epsilon_{ij}(K) = \sum_R t_{ij}(R) e^{iK\cdot R}` is built on
a :math:`37\times37` grid of the supercell Brillouin zone; its :math:`K`
average, the intra-cluster hopping, is moved to ``eloc``. Lattice and local
Hamiltonian are then transformed to the Nambu basis, the latter with a small
d-wave pairing field :math:`\Delta_{ij} = \pm h/\sqrt{2}`, :math:`h = 10^{-3}`,
on the :math:`x` (:math:`+`) and :math:`y` (:math:`-`) bonds, which selects
the symmetry of the order parameter::

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

The filling is fixed with ``fit_mu(mode='fragment')``: the quasiparticle mode
counts the transformed particles and is refused for a Nambu fragment. At the
first filling, :math:`n = 3.6`, the loop starts with ``nseed`` plain mixing
iterations under a larger seed field; every later filling starts from the
previous solution, in two sweeps from :math:`n = 3.6`, up to half filling and
down to :math:`n = 2.9`. One iteration reads::

    lattice.solve_qp([fragment], T=0.0, Tsmearing=Tsmearing)
    fragment.update_hybridization(T=0.0)
    fragment.impose_nambu_spin_symmetry()
    mu = lattice.fit_mu(n_target, [fragment], T=0.0, mu_old=mu,
                        mode='fragment', ntol=1e-8)
    fragment.update_self_energy(T=0.0)
    fragment.impose_nambu_spin_symmetry()

    R_out, Lambda_out = hermitian_gauge(fragment.R, fragment.Lambda)
    x_in, x_out = np.array([R_in, Lambda_in]), np.array([R_out, Lambda_out])
    diis.append(x_out - x_in, x_out)
    fragment.R, fragment.Lambda = diis.solve()

:meth:`~gem.fragment.Fragment.impose_nambu_spin_symmetry` is the Nambu
counterpart of :meth:`~gem.fragment.Fragment.impose_spin_SU2_symmetry`: it
projects the parameters onto those invariant under a :math:`\pi` spin rotation,
which removes magnetization and keeps singlet pairing. With
``paramagnetic = False`` it is not applied, a staggered field is added to the
seed, and the loop finds Néel antiferromagnetism coexisting with d-wave
superconductivity; the results go to ``data_Square_2x2_dwave_U8.0_afm.h5``.

For :math:`n \lesssim 3.4` the paramagnetic d-wave solution is an unstable
fixed point of plain linear mixing, which leaves it and stalls on a state that
is not a solution. The :class:`~gem.diis.DIIS` extrapolation over the last
eight iterates converges to it in about ten iterations. Since DIIS combines
:math:`R` and :math:`\Lambda` of different iterations, they are first brought
to a fixed bath gauge, the one in which :math:`R` is Hermitian (polar
decomposition), which also keeps the spin structure the projection relies on.

Observables
-----------

The anomalous amplitudes
:math:`F_{ij} = \langle c^\dagger_{i\uparrow} c^\dagger_{j\downarrow} \rangle`
are given by :func:`~gem.nambu.nambu_density_matrix`, and the d-wave order
parameter is :math:`\sum_{\langle ij \rangle} \pm F_{ij} / (4\sqrt{2})`.
Since :math:`-\mu N` is :math:`-\mu \sum_a \tau_3^{aa} n'_a` plus a constant
and the interaction has been normal ordered, the physical local energy is::

    eloc_phys = fragment.compute_energy() + mu*np.dot(fragment.charge, np.diag(dm)) + e0

The script stores :math:`\mu`, :math:`Z`, the order parameter, the largest
local moment, the kinetic, potential and total energy, and the converged
parameters for every filling in ``data_Square_2x2_dwave_U8.0.h5`` (or the
``_afm`` file); ``plot_dwave.py`` draws the figure below from both.

Results
-------

Each scan takes about five minutes, and every filling converges to a
residual below :math:`10^{-7}`. Panel (a) shows the d-wave order parameter of
both branches, panel (b) the staggered moment of the AF + d-wave branch.

.. image:: ./images/figE4.png
   :width: 100%
   :align: center

On the pure d-wave branch the order parameter forms a dome with its maximum
near :math:`n = 0.92` per site, and stays finite at half filling. Without the
paramagnetic constraint the two branches coincide for :math:`n \le 0.80`; a
staggered moment appears between :math:`n = 0.80` and :math:`0.825` and grows
to :math:`0.81` at half filling. In this range antiferromagnetism and d-wave
superconductivity coexist, with an energy below the pure d-wave branch by
:math:`0.004` at :math:`n = 0.825` up to :math:`0.48` at half filling. The
antiferromagnetism suppresses the superconductivity near half filling: the
dome peaks lower and earlier (:math:`0.077` at :math:`n = 0.90`), and at half
filling the solution is a Néel insulator with only the field-induced d-wave
response (:math:`2\times10^{-4}`). Along this branch :math:`\mu` decreases
with :math:`n`, a negative compressibility that points to phase separation
between the Néel insulator and a more strongly doped state.

The pure d-wave branch:

.. list-table::
   :header-rows: 1

   * - :math:`n` per site
     - :math:`\mu`
     - :math:`Z`
     - d-wave
     - :math:`E_{tot}`
   * - 0.725
     - 1.1995
     - 0.7519
     - 0.0034
     - -3.7921
   * - 0.775
     - 1.5576
     - 0.7135
     - 0.0079
     - -3.5160
   * - 0.825
     - 1.9169
     - 0.6687
     - 0.0313
     - -3.1686
   * - 0.875
     - 2.1572
     - 0.6222
     - 0.0706
     - -2.7591
   * - 0.925
     - 2.3513
     - 0.6043
     - 0.0799
     - -2.3091
   * - 0.975
     - 2.5901
     - 0.6400
     - 0.0626
     - -1.8176
   * - 1.000
     - 4.0000
     - 0.6587
     - 0.0329
     - -1.5229

The complete tables of both branches and the tests behind this example are
in ``examples/E4/README.md``.
