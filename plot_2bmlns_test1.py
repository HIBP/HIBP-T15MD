# -*- coding: utf-8 -*-
"""
double beamline test
"""

import numpy as np
import matplotlib.pyplot as plt
import hibplib as hb
import hibpplotlib as hbplot

Ebeam = 260.

# %% calculate for analyzer 1
analyzer = 1

runcell('Define Geometry', 'D:/NRCKI/Py/HIBP-T15MD/HIBP-SOLVER.py')
# Load Electric Field
E = {}
# load E for primary beamline
try:
    hb.read_plates('prim', geomT15, E)
    print('\n Primary Beamline loaded')
except FileNotFoundError:
    print('\n Primary Beamline NOT FOUND')

# load E for secondary beamline
try:
    hb.read_plates('sec', geomT15, E)
    print('\n Secondary Beamline loaded')
except FileNotFoundError:
    print('\n Secondary Beamline NOT FOUND')

# %% pass trajectories to analyzer 1
# traj_list = hb.read_traj_list('B1_I1/E100-340_UA23-33_alpha34.0_beta-10.0_x260y-10z1.pkl')
# traj_list_passed = []
# for tr in traj_list:
#     if tr.Ebeam == Ebeam:
#         traj_list_passed.append(tr)

# runcell('Optimize Secondary Beamline', 'D:/NRCKI/py/HIBP-T15MD/HIBP-SOLVER.py')
# traj_list_a3b3_low = copy.deepcopy(traj_list_a3b3)

# %% plot analyzer 1
fig, ax1 = plt.subplots()
hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')
color_sec = 'g'
# plot geometry
geomT15.plot(ax1, axes='XY', plot_analyzer=True)
A2list = []
det_line = np.empty([0, 3])
for tr in traj_list_a3b3_low:
    if tr.Ebeam == Ebeam:
        A2list.append(tr.U['A2'])
        det_line = np.vstack([det_line, tr.RV_sec[0, 0:3]])
        # plot primary
        tr.plot_prim(ax1, axes='XY', color='k', full_primary=False)
        # plot secondary
        tr.plot_sec(ax1, axes='XY', color=color_sec)
ax1.plot(det_line[:, 0], det_line[:, 1], '--o', color=color_sec)

# %% calculate for analyzer 2
analyzer = 2
dUA3 = -dUA3

runcell('Define Geometry', 'D:/NRCKI/Py/HIBP-T15MD/HIBP-SOLVER.py')
# Load Electric Field
E = {}
# load E for primary beamline
try:
    hb.read_plates('prim', geomT15, E)
    print('\n Primary Beamline loaded')
except FileNotFoundError:
    print('\n Primary Beamline NOT FOUND')

# load E for secondary beamline
try:
    hb.read_plates('sec', geomT15, E)
    print('\n Secondary Beamline loaded')
except FileNotFoundError:
    print('\n Secondary Beamline NOT FOUND')

# %% pass trajectories to analyzer 2
# traj_list = hb.read_traj_list('B1_I1/E100-300_UA26-33_alpha34.0_beta-10.0_x260y0z1.pkl')
# traj_list_passed = []
# for tr in traj_list:
#     if tr.Ebeam == Ebeam:
#         traj_list_passed.append(tr)

# runcell('Optimize Secondary Beamline', 'D:/NRCKI/py/HIBP-T15MD/HIBP-SOLVER.py')
# traj_list_a3b3_up = copy.deepcopy(traj_list_a3b3)

# %% plot analyzer 2
color_sec = 'r'
# plot geometry
geomT15.plot(ax1, axes='XY', plot_analyzer=True)
A2list = []
det_line = np.empty([0, 3])
for tr in traj_list_a3b3_up:
    if tr.Ebeam == Ebeam:
        A2list.append(tr.U['A2'])
        det_line = np.vstack([det_line, tr.RV_sec[0, 0:3]])
        # plot primary
        tr.plot_prim(ax1, axes='XY', color='k', full_primary=False)
        # plot secondary
        tr.plot_sec(ax1, axes='XY', color=color_sec)
ax1.plot(det_line[:, 0], det_line[:, 1], '--o', color=color_sec)

ax1.set_title('E={} keV, Btor={} T, Ipl={} MA'
              .format(Ebeam, Btor, Ipl))

plt.show()
