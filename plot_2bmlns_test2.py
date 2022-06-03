# -*- coding: utf-8 -*-
"""
double beamline test GRIDS
"""

import numpy as np
import matplotlib.pyplot as plt
import hibplib as hb
import hibpplotlib as hbplot


# %%
def plot_grid(ax, traj_list, geom, Btor, Ipl, onlyE=False,
              col='k', linestyle_A2='--', linestyle_E='-',
              marker_A2='*', marker_E='p'):
    '''
    plot detector grid in XY planes using axes ax
    '''
    # plot geometry
    geom.plot(ax1, axes='XY')

    # get the list of A2 and Ebeam
    A2list = []
    Elist = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].U['A2'])
        Elist.append(traj_list[i].Ebeam)

    # make sorted arrays of non repeated values
    A2list = np.unique(A2list)
    N_A2 = A2list.shape[0]
    Elist = np.unique(Elist)
    N_E = Elist.shape[0]

    E_grid = np.full((N_A2, 3, N_E), np.nan)
    A2_grid = np.full((N_E, 3, N_A2), np.nan)

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))
    # set title
    ax.set_title('Eb = [{}, {}] keV, UA2 = [{}, {}] kV,'
                 ' Btor = {} T, Ipl = {} MA'
                 .format(traj_list[0].Ebeam, traj_list[-1].Ebeam, UA2_min,
                         UA2_max, Btor, Ipl))

    # make a grid of constant E
    for i_E in range(0, N_E, 1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].Ebeam == Elist[i_E]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_sec[0, 0]
                y = traj_list[i_tr].RV_sec[0, 1]
                z = traj_list[i_tr].RV_sec[0, 2]
                E_grid[k, :, i_E] = [x, y, z]

        ax.plot(E_grid[:, 0, i_E], E_grid[:, 1, i_E], color=col,
                linestyle=linestyle_E,
                marker=marker_E,
                label=str(int(Elist[i_E]))+' keV')

    if onlyE:
        ax.legend()
        return 0
    # make a grid of constant A2
    for i_A2 in range(0, N_A2, 1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].U['A2'] == A2list[i_A2]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_sec[0, 0]
                y = traj_list[i_tr].RV_sec[0, 1]
                z = traj_list[i_tr].RV_sec[0, 2]
                A2_grid[k, :, i_A2] = [x, y, z]

        ax1.plot(A2_grid[:, 0, i_A2], A2_grid[:, 1, i_A2], color=col,
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2], 1))+' kV')

    # ax.legend()
    # ax.set(xlim=(0.9, 4.28), ylim=(-1, 1.5), autoscale_on=False)
    plt.show()

# %% calculate for analyzer 1
analyzer = 1
dUA3 = abs(dUA3)

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
traj_list = hb.read_traj_list('B1_I1/E100-340_UA22-34_alpha34.0_beta-10.0_x260y-10z1.pkl')

traj_list_passed = copy.deepcopy(traj_list)
runcell('Optimize Secondary Beamline', 'D:/NRCKI/py/HIBP-T15MD/HIBP-SOLVER.py')
traj_list_a3b3_low = copy.deepcopy(traj_list_a3b3)

# %% plot grid for analyzer 1
fig, ax1 = plt.subplots()
hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')
color_sec = 'g'
plot_grid(ax1, traj_list_a3b3_low, geomT15, Btor, Ipl,
          onlyE=False, col=color_sec)

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
traj_list = hb.read_traj_list('B1_I1/E100-300_UA24-34_alpha34.0_beta-10.0_x260y0z1.pkl')
traj_list_passed = []

traj_list_passed = copy.deepcopy(traj_list)
runcell('Optimize Secondary Beamline', 'D:/NRCKI/py/HIBP-T15MD/HIBP-SOLVER.py')
traj_list_a3b3_up = copy.deepcopy(traj_list_a3b3)

# %% plot analyzer 2
color_sec = 'r'
plot_grid(ax1, traj_list_a3b3_up, geomT15, Btor, Ipl,
          onlyE=False, col=color_sec)
