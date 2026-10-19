######################################################################
# Plot of Example 4: d-wave order parameter as a function of filling,
# from the output of Square_2x2_dwave_sc.py.
######################################################################

import numpy as np
import h5py
import matplotlib.pyplot as plt

U, Nc = 8.0, 4
h5_file = f'data_Square_2x2_dwave_U{U}.h5'

with h5py.File(h5_file, 'r') as h5f:
    data = sorted((float(g[1:]), float(h5f[g]['dwave'][()])) for g in h5f)
n, dwave = np.array(data).T

surface, ink, ink2, grid, series = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6'
fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor=surface)
ax.set_facecolor(surface)
ax.plot(n/Nc, dwave, '-o', color=series, lw=2, ms=8, mec=surface, mew=2)

ax.set_xlabel('filling $n$ per site', color=ink2)
ax.set_ylabel('d-wave order parameter', color=ink2)
ax.set_title(f'2x2 cluster RISB, $U={U:g}|t|$', color=ink, loc='left')
ax.set_ylim(bottom=0)
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
