# E4 — d-wave superconductivity in the 2x2 cluster Hubbard model

`Square_2x2_dwave_sc.py` computes the d-wave superconducting order parameter
of the doped Hubbard model on the square lattice. It uses a 2x2 cluster
fragment at B = 1 (cluster RISB), with U = 8|t|, at fillings from n = 2.9 to
4.0 electrons per cluster. It shows the three ingredients of a
superconducting calculation in GEM:

* the Nambu basis of `gem.nambu`;
* the non-magnetic projection `Fragment.impose_nambu_spin_symmetry`;
* DIIS extrapolation (`gem.diis`).

![d-wave order parameter as a function of filling](../../doc/examples/images/figE4.png)

## Running

```bash
PYTHONPATH=/path/to/gem/python python3 Square_2x2_dwave_sc.py
python3 plot_dwave.py
```

The scan takes about 5 minutes on a laptop: about 30 iterations at the first
filling and about 10 at each of the others. The embedding problem has 16
levels, and the S_z = 0 sector has 12870 states. Results go to
`data_Square_2x2_dwave_U8.0.h5`, with one group `n<n>` per filling holding:

| dataset | meaning |
|---|---|
| `mu` | chemical potential |
| `Z` | (R^dag R)_00, quasiparticle weight of site 0, spin up |
| `dwave` | abs d-wave order parameter (see below) |
| `mag` | largest local moment abs(n_up - n_dn), zero for the paramagnetic solution |
| `Ekin`, `Epot`, `Etot` | kinetic, interaction and total energy per cluster, without -mu N |
| `R`, `Lambda`, `denMat` | converged parameters and impurity density matrix, in the Nambu basis |

`plot_dwave.py` reads that file and writes `figE4.png`, the figure above.

## Model

The 2x2 supercell has sites 0 (0,0), 1 (1,0), 2 (0,1) and 3 (1,1), and nearest-neighbour
hopping t = -1. The cluster dispersion e_ij(K) = sum_R t_ij(R) exp(iK.R)
is built on a 37x37 grid of the supercell Brillouin zone. Its K average,
the intra-cluster hopping, is moved to the local Hamiltonian `eloc`. The
interaction is U n_up n_dn on every site.

A small d-wave pairing field (`field = 1e-3`) on the x bonds (+) and the
y bonds (-) selects the symmetry of the order parameter:

    Delta_ij (c^dag_{i,up} c^dag_{j,dn} - c^dag_{i,dn} c^dag_{j,up}) + h.c.,
    Delta_ij = +-field/sqrt(2)

The order parameter is computed from the anomalous amplitudes
F_ij = <c^dag_{i,up} c^dag_{j,dn}> as sum_<ij> (+-) F_ij / (4 sqrt 2).

## How superconductivity is treated

GEM solves particle-number conserving problems. Singlet pairing is brought
to that form by the particle-hole transformation of the spin-down orbitals,
c_{k,dn} -> d^dag_{-k,dn} (`gem.nambu`):

| physical | Nambu basis |
|---|---|
| hopping, spin up | unchanged |
| hopping, spin down | hole block -e_dn(k) (real hoppings) |
| pairing Delta c^dag_up c^dag_dn | hybridization Delta c^dag_up d |
| U n_up n_dn | -U n_up n_d + U n_up |
| -mu N | -mu tau_3 + const |
| conserved N | physical S_z |

`Lattice`, `Fragment` and `SimpleED` are then used unchanged:

```python
eks_n, _ = nambu_one_body(eks)                                   # lattice
eloc_n, U_n, e0 = nambu_local(eloc, Utensor, pairing=Delta)      # local part
solver = SimpleED(ntot, use_Ntot=True, N_sector=ntot//2)         # S_z = 0, no use_Sz
fragment = Fragment(nimp, nbath, eloc_n, U_n, solver, nambu=True)
```

* With `nambu=True` the chemical potential couples to tau_3, and
  `fragment.nfill` is the physical filling. The filling is fixed with
  `lattice.fit_mu(..., mode='fragment')`. The quasiparticle mode counts the
  transformed particles, i.e. S_z, and raises an error for a Nambu fragment.
* `nambu_density_matrix(fragment.denMat[:nimp, :nimp])` returns the physical
  rho_up, rho_dn and F.
* Since -mu N and the interaction have been rewritten, the physical local
  energy is
  `fragment.compute_energy() + mu*np.dot(fragment.charge, np.diag(dm)) + e0`.

Only singlet (opposite-spin) pairing can be represented. Spin-flip terms and
equal-spin pairing raise a `ValueError`.

## Self-consistency loop

```python
for it in range(itmax):
    lattice.solve_qp([fragment], T=0.0, Tsmearing=Tsmearing)
    fragment.update_hybridization(T=0.0)
    fragment.impose_nambu_spin_symmetry()
    mu = lattice.fit_mu(n_target, [fragment], T=0.0, mu_old=mu, mode='fragment')
    fragment.update_self_energy(T=0.0)
    fragment.impose_nambu_spin_symmetry()
    R_out, Lambda_out = hermitian_gauge(fragment.R, fragment.Lambda)
    diis.append([R_out, Lambda_out] - [R_in, Lambda_in], [R_out, Lambda_out])
    fragment.R, fragment.Lambda = diis.solve()
```

(The `append` line is schematic: the script stacks the pairs into arrays.)

**Seed.** At the first filling (n = 3.6) the loop starts with 20 plain mixing
iterations under a larger pairing field (`seed_field = 0.05`). Every later
filling starts from the previous solution. The scan runs in two sweeps from
n = 3.6, one up to half filling and one down to n = 2.9.

**Paramagnetic projection.** `impose_nambu_spin_symmetry` is the Nambu
counterpart of `impose_spin_SU2_symmetry`. A pi spin rotation acts on the
Nambu spinor as psi -> K psi^dag, with K = diag([[0,1],[-1,0]], ...). The
projection keeps the invariant parameters, R = K^T R^* K, D = K^T D^* K,
Lambda = -K^T Lambda^T K and Lambda_c = -K^T Lambda_c^T K, which removes
S_z magnetization and keeps singlet pairing. Without it
(`paramagnetic = False`) the loop flows to Neel antiferromagnetism coexisting
with d-wave pairing near half filling. That state has a lower energy: at
n = 3.6 the staggered moment is 0.52 and Etot = -2.658, against -2.539 for
the paramagnetic state.

**DIIS.** The paramagnetic d-wave solution is an unstable fixed point of the
plain mixing loop for n <~ 3.4. Linear mixing leaves it: the deviation grows
by about 1.5 per iteration. It then stalls on a state that is not a solution:
mu and the order parameter freeze, but the residual stays at about 0.1. DIIS
(Pulay) extrapolation over the last 8 iterates converges to it in about 10
iterations. DIIS combines R and Lambda from different iterations, which is
only meaningful in a fixed bath gauge (R -> U R, Lambda -> U Lambda U^dag).
The script uses the gauge in which R is Hermitian (polar decomposition, B = 1),
which also preserves the spin structure that the projection relies on.

## Results

| n per cluster | n per site | mu | Z | d-wave | Etot |
|---|---|---|---|---|---|
| 2.9 | 0.725 | 1.1995 | 0.7519 | 0.0034 | -3.7921 |
| 3.0 | 0.750 | 1.3851 | 0.7333 | 0.0051 | -3.6634 |
| 3.1 | 0.775 | 1.5576 | 0.7135 | 0.0079 | -3.5160 |
| 3.2 | 0.800 | 1.7357 | 0.6919 | 0.0155 | -3.3515 |
| 3.3 | 0.825 | 1.9169 | 0.6687 | 0.0313 | -3.1686 |
| 3.4 | 0.850 | 2.0532 | 0.6441 | 0.0536 | -2.9698 |
| 3.5 | 0.875 | 2.1572 | 0.6222 | 0.0706 | -2.7591 |
| 3.6 | 0.900 | 2.2486 | 0.6088 | 0.0786 | -2.5391 |
| 3.7 | 0.925 | 2.3513 | 0.6043 | 0.0799 | -2.3091 |
| 3.8 | 0.950 | 2.4534 | 0.6141 | 0.0756 | -2.0689 |
| 3.9 | 0.975 | 2.5901 | 0.6400 | 0.0626 | -1.8176 |
| 4.0 | 1.000 | 4.0000 | 0.6587 | 0.0329 | -1.5229 |

The order parameter forms a dome with its maximum near n = 0.92 per site.
At half filling it does not vanish: the remnant is induced by the pairing
field. All points converge to a residual below 1e-7 with zero local moment.

## Tests

`test/python/test_nambu.py` and `test/python/test_diis.py` (run by
`make test`) check the pieces this example relies on:

| test | what it checks |
|---|---|
| `test_spectrum_kanamori` | Brute-force Jordan-Wigner check in the full Fock space: H_loc with complex hopping, Kanamori U, J and a random singlet pairing field has the same spectrum as its Nambu transform plus `e0`. |
| `test_hubbard` | U n_up n_dn maps to -U n_up n_d + U n_up. |
| `test_dispersion_hole_block` | The hole block of a k-resolved dispersion is -e_dn(k), and the constant is Tr e_dn(k). |
| `test_refuse_sz_breaking` | Spin-flip hopping and S_z-breaking interactions raise `ValueError`. |
| `test_fragment_options` | `nambu=True` sets the tau_3 charge, refuses `impose_spin_SU2_symmetry` and the quasiparticle `fit_mu`. |
| `test_bcs_noninteracting` | End to end: square lattice, U = 0, local s-wave field, T > 0, filling fitted with `fit_mu(mode='fragment')`. RISB is exact here, so the converged fragment must reproduce the BdG k-sum for F, n and rho_up = rho_dn, and be invariant under `impose_nambu_spin_symmetry`. |
| `test_nambu_spin_projector` | `impose_nambu_spin_symmetry` is idempotent, keeps Lambda Hermitian and removes the uniform magnetization term from Lambda. |
| `test_mixing_diverges` / `test_unstable_fixed_point` | On a linear map with an eigenvalue 1.5, linear mixing diverges while DIIS converges to the fixed point. |
| `test_coefficients_and_history` | DIIS coefficients are real and sum to one, so Hermitian states stay Hermitian; the history is capped at `nvec`. |

Run them alone with

```bash
PYTHONPATH=python python3 test/python/test_nambu.py
PYTHONPATH=python python3 test/python/test_diis.py
```
