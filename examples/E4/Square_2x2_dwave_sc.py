######################################################################
# Example 4
# d-wave superconductivity in the doped Hubbard model on the square
# lattice, described with a 2x2 cluster fragment (cluster RISB, B=1).
# Singlet pairing is treated with the Nambu (spin-down particle-hole)
# transformation of gem.nambu: the BdG problem becomes a particle-number
# conserving one and the usual GEM loop is used unchanged. The filling is
# fixed with fit_mu(mode='fragment'); a small d-wave field selects the
# symmetry of the order parameter. With paramagnetic=True the parameters are
# projected onto the non-magnetic ones at every iteration: without it the
# loop flows to antiferromagnetism coexisting with d-wave superconductivity.
######################################################################

import numpy as np
from gem.fragment import Fragment
from gem.lattice import Lattice
from gem.solvers.simple_ed import SimpleED
from gem.nambu import nambu_one_body, nambu_local, nambu_density_matrix, nambu_interaction
import h5py

# 2x2 cluster:   2 3
#                0 1
sites = [(0, 0), (1, 0), (0, 1), (1, 1)]
Nc = len(sites)
nimp, B = 2*Nc, 1
nbath = nimp * B
ntot = nimp + nbath

# Physical Parameters (n_list: electrons per cluster)
U, t, field, seed_field = 8.0, -1.0, 1e-3, 5e-2
n_list = [3.6, 3.7, 3.8, 3.9, 4.0]

paramagnetic = True

# Self-consistency Parameters
itmax, tol, mix, nseed, Tsmearing, nk = 500, 1e-6, 0.3, 20, 1e-5, 37
h5_file = f'data_Square_2x2_dwave_U{U}.h5'


def cluster_dispersion(nk, t):
    '''Hopping of the 2x2 supercell, e_ij(K) = sum_R t_ij(R) exp(iK.R).'''
    ks = 2*np.pi*np.arange(nk)/nk
    eks = np.zeros((nk*nk, Nc, Nc), dtype=np.complex128)
    for ik, (k1, k2) in enumerate((k1, k2) for k1 in ks for k2 in ks):
        for i, (x, y) in enumerate(sites):
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                X, Y = x+dx, y+dy
                j = sites.index((X % 2, Y % 2))
                eks[ik, i, j] += t*np.exp(1j*(k1*(X//2) + k2*(Y//2)))
    return eks


# Spinful dispersion; its k average (intra-cluster hopping) goes to eloc
eks = np.kron(cluster_dispersion(nk, t), np.eye(2))
eloc = eks.mean(axis=0)
eks -= eloc

# Interaction tensor: U n_up n_dn on every site
Utensor = np.zeros((nimp, nimp, nimp, nimp))
for i in range(Nc):
    Utensor[2*i, 2*i, 2*i+1, 2*i+1] = U
    Utensor[2*i+1, 2*i+1, 2*i, 2*i] = U

# d-wave form factor: +1 on x bonds (0-1, 2-3), -1 on y bonds (0-2, 1-3)
dwave = np.zeros((Nc, Nc))
for i, j, s in [(0, 1, 1), (2, 3, 1), (0, 2, -1), (1, 3, -1)]:
    dwave[i, j] = dwave[j, i] = s


def dwave_order(F):
    '''d-wave order parameter from F_ij = <c^dag_{i up} c^dag_{j dn}>.'''
    return np.sum(dwave*F)/(4*np.sqrt(2))


# Nambu transformation of lattice and local Hamiltonian.
# The pairing field d*(c^dag_{i up} c^dag_{j dn} - c^dag_{i dn} c^dag_{j up}) on
# every bond has amplitude Delta_ij = Delta_ji = field/sqrt(2)*dwave_ij.
eks_n, _ = nambu_one_body(eks)
lattice = Lattice(eks_n)
eloc_n, U_n, e0 = nambu_local(eloc, Utensor, pairing=field/np.sqrt(2)*dwave)
eloc_seed, _, _ = nambu_local(eloc, Utensor, pairing=seed_field/np.sqrt(2)*dwave)
_, h_U, e_U = nambu_interaction(Utensor)

# N_sector=ntot//2: half-filled embedding <=> physical S_z = 0
edsolver = SimpleED(ntot, use_Ntot=True, N_sector=ntot//2)
fragment = Fragment(nimp, nbath, eloc_n, U_n, edsolver, nambu=True)


with h5py.File(h5_file, 'a') as _:
    pass

mu = U/2
for i_n, n_target in enumerate(n_list):
    print('--------------------------------------------------------')
    print(f'GEM loop started with U={U} and n={n_target}')

    gap = 0.0
    for it in range(itmax):
        lattice.solve_qp([fragment], T=0.0, Tsmearing=Tsmearing)
        fragment.update_hybridization(T=0.0)
        if paramagnetic:
            fragment.impose_nambu_spin_symmetry()

        # a larger d-wave field for the first iterations drives the system
        # to the superconducting solution
        fragment.eloc = eloc_seed if (i_n == 0 and it < nseed) else eloc_n
        mu = lattice.fit_mu(n_target, [fragment], T=0.0, mu_old=mu,
                            mode='fragment', ntol=1e-8)

        Lambda_old, R_old = fragment.Lambda.copy(), fragment.R.copy()
        fragment.update_self_energy(T=0.0)
        if paramagnetic:
            fragment.impose_nambu_spin_symmetry()
        diff_LR = max(np.abs(fragment.Lambda - Lambda_old).max(),
                      np.abs(fragment.R - R_old).max())
        fragment.Lambda = (1 - mix)*fragment.Lambda + mix*Lambda_old
        fragment.R = (1 - mix)*fragment.R + mix*R_old

        gap_old = gap
        _, _, F = nambu_density_matrix(fragment.denMat[:nimp, :nimp])
        gap = dwave_order(F)
        diff = max(diff_LR, abs(gap - gap_old))
        print(f'iteration: {it}  diff={diff:.2e}  mu={mu:.6f}  '
              f'n={fragment.nfill.real:.6f}  d-wave={gap.real:.6f}')
        if (diff < tol and it > nseed) or it == itmax - 1:
            print(f'--- Exiting after {it}/{itmax} - diff={diff:.2e} ---')
            break

    # Observables
    dm = fragment.denMat[:nimp, :nimp]
    rho_up, rho_dn, F = nambu_density_matrix(dm)
    mag = np.max(np.abs(np.diag(rho_up - rho_dn)))
    Z = (fragment.R.conj().T @ fragment.R)[0, 0].real
    ekin = lattice.compute_ekin([fragment], T=0.0, Tsmearing=Tsmearing).real
    # <H_loc> without -mu N: remove the -mu tau3 of the embedding, add the constant
    eloc_phys = (fragment.compute_energy() + mu*np.dot(fragment.charge, np.diag(dm)) + e0).real
    epot = (fragment.E2loc + np.sum(h_U*dm) + e_U).real
    etot = ekin + eloc_phys

    print('--------------------------------------------------------')
    print(f'GEM loop ended with U={U} and n={n_target}')
    print(f'mu={mu:.6f} Z={Z:.6f} |d-wave|={abs(gap):.6f} |m|={mag:.6f} '
          f'Ekin={ekin:.6f} Epot={epot:.6f} Etot={etot:.6f}')
    print('--------------------------------------------------------')

    with h5py.File(h5_file, 'a') as h5f:
        grp_name = f'n{n_target:.3f}'
        if grp_name in h5f:
            del h5f[grp_name]
        grp = h5f.create_group(grp_name)
        for key, val in [('mu', mu), ('Z', Z), ('dwave', abs(gap)), ('mag', mag),
                         ('Ekin', ekin), ('Epot', epot), ('Etot', etot),
                         ('R', fragment.R), ('Lambda', fragment.Lambda),
                         ('denMat', dm)]:
            grp.create_dataset(key, data=val)
