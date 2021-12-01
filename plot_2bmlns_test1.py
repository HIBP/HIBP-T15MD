# -*- coding: utf-8 -*-
"""
double beamline test
"""

import numpy as np
import matplotlib.pyplot as plt
import hibplib as hb
import hibplotlib as hbplot

fig, ax1 = plt.subplots()
hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')

# calculate for analyzer 1
analyzer = 1

runcell('Define Geometry', 'D:/Philipp/Py/HIBP-T15MD/HIBP-SOLVER.py')
runcell('Load Electric Field', 'D:/Philipp/Py/HIBP-T15MD/HIBP-SOLVER.py')

color_sec = 'r'
# plot geometry
geomT15.plot(ax1, axes='XY', plot_analyzer=False)
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

# calculate for analyzer 2
analyzer = 2

runcell('Define Geometry', 'D:/Philipp/Py/HIBP-T15MD/HIBP-SOLVER.py')
runcell('Load Electric Field', 'D:/Philipp/Py/HIBP-T15MD/HIBP-SOLVER.py')

color_sec = 'g'
# plot geometry
geomT15.plot(ax1, axes='XY', plot_analyzer=False)
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
