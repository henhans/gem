# E4 — d-wave superconductivity in the 2x2 cluster Hubbard model

`Square_2x2_dwave_sc.py` computes the d-wave superconducting order parameter
of the doped square-lattice Hubbard model at U = 8|t| with a 2x2 cluster
fragment at B = 1 (cluster RISB), for a list of fillings n (electrons per
cluster). It is the reference calculation for the superconducting support
added in `gem.nambu`, and it reproduces the original `simple_ga`
implementation (see [Validation](#validation)).

## Running

```bash
PYTHONPATH=/path/to/gem/python python3 Square_2x2_dwave_sc.py
```

The first filling takes about 2 minutes on a laptop (about 45 iterations; the
embedding problem has 16 levels, 12870 states in the S_z = 0 sector), and
the following ones less, since each starts from the previous solution.
Results go to `data_Square_2x2_dwave_U8.0.h5`, one group `n<n>` per filling
holding `mu`, `Z`, `dwave`, `mag` (largest local moment), `Ekin`, `Epot`,
`Etot`, `R`, `Lambda` and the impurity `denMat` (Nambu basis).

## How superconductivity is treated

GEM only solves particle-number conserving problems. Singlet pairing is
brought to that form by the particle-hole transformation of the spin-down
orbitals, c_{k,dn} -> d^dag_{-k,dn} (`gem.nambu`):

| physical | Nambu basis |
|---|---|
| hopping, spin up | unchanged |
| hopping, spin down | hole block -e_dn(k) (real hoppings) |
| pairing D c^dag_up c^dag_dn | hybridization D c^dag_up d |
| U n_up n_dn | -U n_up n_d + U n_up |
| -mu N | -mu tau_3 + const |
| conserved N | physical S_z |

So `Lattice`, `Fragment` and `SimpleED` are used unchanged, with three
settings:

```python
eks_n, _ = nambu_one_body(eks)                                   # lattice
eloc_n, U_n, e0 = nambu_local(eloc, Utensor, pairing=Delta)      # local part
solver = SimpleED(ntot, use_Ntot=True, N_sector=ntot//2)         # S_z = 0, no use_Sz
fragment = Fragment(nimp, nbath, eloc_n, U_n, solver, nambu=True)
```

The anomalous amplitudes F_ij = <c^dag_{i,up} c^dag_{j,dn}> come from
`nambu_density_matrix(fragment.denMat[:nimp, :nimp])`.

### Paramagnetic constraint

`paramagnetic = True` (the default) calls
`fragment.impose_nambu_spin_symmetry()` after both updates. It is the Nambu
counterpart of `impose_spin_SU2_symmetry`: a pi spin rotation acts on the
Nambu spinor as psi -> K psi^dag, with K = diag([[0,1],[-1,0]], ...), and the
parameters are projected onto R = K^T R^* K, D = K^T D^* K,
Lambda = -K^T Lambda^T K and Lambda_c = -K^T Lambda_c^T K. This removes S_z
magnetization and keeps singlet pairing. `simple_ga` imposes the same
constraint through its spin-symmetric basis for R and Lambda.

Without the projector (`paramagnetic = False`) the paramagnetic d-wave
solution is unstable near half filling. The loop passes close to it and then
flows to Neel antiferromagnetism coexisting with d-wave pairing, which has a
lower energy. At n = 3.6: staggered moment 0.52, mu = 1.662, d-wave 0.077,
Etot = -2.658, against -2.539 for the paramagnetic state. The same instability
shows at fixed mu, where the loop drifts to half filling.

## Tests

`test/python/test_nambu.py` (run by `make test`) checks the pieces this
example relies on:

| test | what it checks |
|---|---|
| `test_spectrum_kanamori` | Brute-force Jordan-Wigner check, full Fock space: H_loc with complex hopping, Kanamori U, J and a random singlet pairing field has the same spectrum as its Nambu transform plus `e0`. |
| `test_hubbard` | U n_up n_dn maps to -U n_up n_d + U n_up. |
| `test_dispersion_hole_block` | The hole block of a k-resolved dispersion is -e_dn(k), and the constant is Tr e_dn(k). |
| `test_refuse_sz_breaking` | Spin-flip hopping and S_z-breaking interactions raise `ValueError`. |
| `test_fragment_options` | `nambu=True` sets the tau_3 charge, refuses `impose_spin_SU2_symmetry` and the quasiparticle `fit_mu`. |
| `test_bcs_noninteracting` | End to end: square lattice, U = 0, local s-wave field, T > 0, filling fitted with `fit_mu(mode='fragment')`. RISB is exact here, so the converged fragment must reproduce the BdG k-sum for F, n and rho_up = rho_dn, and be invariant under `impose_nambu_spin_symmetry`. |
| `test_nambu_spin_projector` | `impose_nambu_spin_symmetry` is idempotent, keeps Lambda Hermitian and removes the uniform magnetization term from Lambda. |

Run them alone with

```bash
PYTHONPATH=python python3 test/python/test_nambu.py
```

## Validation

The same model (37x37 supercell k grid, d-wave field 1e-3, T = 1e-5
smearing) was solved with the full-Nambu cluster-RISB code `simple_ga`
(`examples/clusters/2D/Nc4/sc_run/U8_scan_n_up`). The cluster dispersion
built by this script agrees with the one from `simple_tb` to 2e-15.

Both columns are converged to about 1e-6, and `simple_ga` uses `xtol` 1e-4.
Eloc = Eloc1 + Eloc2 in `simple_ga`, and Etot - Ekin here.

| n | code | mu | Z | d-wave | Epot | Eloc |
|---|---|---|---|---|---|---|
| 3.6 | GEM | 2.248576 | 0.608811 | 0.078641 | 1.404304 | -1.113781 |
| | simple_ga | 2.248577 | 0.608811 | 0.078641 | 1.404305 | -1.113780 |
| 3.7 | GEM | 2.351307 | 0.604291 | 0.079882 | 1.399232 | -1.116650 |
| | simple_ga | 2.351306 | 0.604292 | 0.079882 | 1.399232 | -1.116651 |
| 3.8 | GEM | 2.453406 | 0.614052 | 0.075590 | 1.389704 | -1.108229 |
| | simple_ga | 2.453397 | 0.614040 | 0.075590 | 1.389704 | -1.108232 |
| 3.9 | GEM | 2.590130 | 0.639968 | 0.062587 | 1.364560 | -1.096924 |
| | simple_ga | 2.590130 | 0.639968 | 0.062572 | 1.364530 | -1.096917 |
| 4.0 | GEM | 4.000000 | 0.658724 | 0.032890 | 1.363936 | -1.047768 |
| | simple_ga | 4.000014 | 0.658717 | 0.032890 | 1.363940 | -1.047765 |

The local magnetic moment is zero at every filling.

Two notes on comparing with `simple_ga` output:

* **Kinetic energy.** `simple_ga` reports Ek twice the value here (-2.850667
  against -1.425333 at n = 3.6), because `calc_Ek_T` sums over the full
  2x(2Nc) BdG matrix and so counts each spin twice. Its local energies
  agree with GEM. The `Etot` column in `simple_ga`'s
  `n_mu_Z_gaps_Ek_Eloc1_Eloc2_Etot` files therefore contains the double
  counted kinetic energy; GEM's `Etot` does not.
* **Hole block.** `simple_ga` builds the BdG hole block as `-ek.T`, which is
  -e(-k). The correct block for local pairing is -e(-k)^T, which is -e(k).
  For this cluster the two are related by inversion (0<->3, 1<->2), which
  maps the d-wave field onto minus itself, so the spectra agree (to 4e-15)
  and so do all observables. For a pairing field that is not
  inversion-symmetric the two conventions differ.
