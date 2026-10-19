######################################################################
# Plot of Example 4: d-wave order parameter as a function of filling,
# from the output of Square_2x2_dwave_sc.py. If the AF+SC branch
# (paramagnetic=False) has been computed too, it is compared with the
# pure d-wave branch: its staggered moment is shown in a second panel and
# the energy difference between the two branches in a third one.
######################################################################

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

U, Nc = 8.0, 4
files = {'pure d-wave (paramagnetic)': f'data_Square_2x2_dwave_U{U}.h5',
         'AF + d-wave': f'data_Square_2x2_dwave_U{U}_afm.h5'}
colors = {'pure d-wave (paramagnetic)': '#2a78d6', 'AF + d-wave': '#eb6834'}


def load(h5_file):
    with h5py.File(h5_file, 'r') as h5f:
        data = sorted((float(g[1:]), float(h5f[g]['dwave'][()]), float(h5f[g]['mag'][()]),
                       float(h5f[g]['Etot'][()])) for g in h5f)
    return np.array(data).T


branches = {name: load(f) for name, f in files.items() if os.path.exists(f)}
both = len(branches) > 1

surface, ink, ink2, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
fig, axes = plt.subplots(1, 3 if both else 1, figsize=(15, 4.2) if both else (6.4, 4.2),
                         facecolor=surface, squeeze=False)
axes = axes[0]
for name, (n, dwave, mag, etot) in branches.items():
    axes[0].plot(n/Nc, dwave, '-o', color=colors[name], lw=2, ms=8, mec=surface, mew=2,
                 label=name)
    if both and name == 'AF + d-wave':
        axes[1].plot(n/Nc, mag, '-o', color=colors[name], lw=2, ms=8, mec=surface, mew=2)

axes[0].set_ylabel('d-wave order parameter', color=ink2)
axes[0].set_title(f'(a) 2x2 cluster RISB, $U={U:g}|t|$', color=ink, loc='left')
if both:
    axes[0].legend(frameon=False, labelcolor=ink2, loc='upper left')
    axes[1].set_ylabel('staggered moment $|n_\\uparrow - n_\\downarrow|$', color=ink2)
    axes[1].set_title('(b) AF + d-wave branch', color=ink, loc='left')
    n_pm, _, _, e_pm = branches['pure d-wave (paramagnetic)']
    n_af, _, _, e_af = branches['AF + d-wave']
    assert np.allclose(n_pm, n_af), 'the two branches must have the same fillings'
    axes[2].axhline(0.0, color=ink2, lw=1)
    axes[2].plot(n_af/Nc, e_af - e_pm, '-o', color=colors['AF + d-wave'], lw=2, ms=8,
                 mec=surface, mew=2)
    axes[2].set_ylabel('$E_{tot}$(AF + d-wave) $-$ $E_{tot}$(pure d-wave)', color=ink2)
    axes[2].set_title('(c) energy difference per cluster', color=ink, loc='left')
for ax in axes[:2]:
    ax.set_ylim(bottom=0)
for ax in axes:
    ax.set_facecolor(surface)
    ax.set_xlabel('filling $n$ per site', color=ink2)
    ax.grid(color=grid, lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2, length=0)
fig.tight_layout()
fig.savefig('figE4.png', dpi=200)
print('Written figE4.png')
