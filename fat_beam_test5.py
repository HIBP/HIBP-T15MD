import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import alphashape
import sys

# %%
''' test FAT beam with focusing
'''
Ebeam = 140.
UA2 = 3.0

n_slits = 7
# add slits to Geometry
geomT15.add_slits(n_slits=n_slits, slit_dist=0.01, slit_w=5e-3,
                  slit_l=0.1, slit_gamma=-20.)
r_slits = geomT15.slits_edges
rs = geomT15.r_dict['slit']
# calculate normal to slit plane
slit_plane_n = geomT15.slit_plane_n

# %%
traj_list_copy = copy.deepcopy(traj_list_oct)
# traj_list_copy = copy.deepcopy(traj_list_passed)

# %%
# set number of filaments in a beam
d_beam = 0.02  # beam diameter [m]
n_filaments_xy = 7  # number of filaments in xy plane (must be ODD)
skip_center_traj = True
n_gamma = 4  # number of chords in beam cross-section
foc_len = 50  # distance from the first point of the trajectory to the focus
drad = np.pi/180.  # converts degrees to radians

fat_beam_list = []

for tr in traj_list_copy:
    if tr.Ebeam == Ebeam and tr.U[0] == UA2:
        print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U[0]))
    else:
        continue
    r0 = tr.RV0[0, :3]
    v0_abs = np.linalg.norm(tr.RV0[0, 3:])
    for i in range(n_filaments_xy):
        # skip center traj
        if abs(i - (n_filaments_xy-1)/2) < 1e-6 and skip_center_traj:
            continue
        # beam convergence angle
        alpha_conv = np.arctan((i - (n_filaments_xy-1)/2) *
                               (d_beam/(n_filaments_xy-1)) / foc_len)
        # set coords and velocity at the center of coord system
        x = 0.0
        y = -(i - (n_filaments_xy-1)/2) * (d_beam/(n_filaments_xy-1))
        z = 0.0
        r = np.array([x, y, z])
        print('XY filament number = {}'.format(i+1))
        v0 = v0_abs * np.array([-np.cos(alpha_conv),
                                np.sin(alpha_conv), 0.])
        # for gamma in np.arange(np.pi/n_gamma, np.pi, np.pi/n_gamma):
        for gamma in np.arange(0, np.pi, np.pi/n_gamma):
            gamma = gamma/drad
            print('gamma = ', gamma)
            # rotate and translate r0 to beam starting point
            r_rot = hb.rotate(r, axis=(1, 0, 0), deg=gamma)
            r_rot = hb.rotate(r_rot, axis=(0, 0, 1), deg=tr.alpha)
            r_rot = hb.rotate(r_rot, axis=(0, 1, 0), deg=tr.beta)
            r_rot += r0
            v_rot = hb.rotate(v0, axis=(1, 0, 0), deg=gamma)
            v_rot = hb.rotate(v_rot, axis=(0, 0, 1), deg=tr.alpha)
            v_rot = hb.rotate(v_rot, axis=(0, 1, 0), deg=tr.beta)

            tr_fat = copy.deepcopy(tr)
            tr_fat.RV0[0, :] = np.hstack([r_rot, v_rot])
            # tr_fat.U = [0., 0., 0., 0.]
            # tr_fat.pass_prim(E, B, geomT15, tmax=0.01)
            tr_fat = hb.pass_to_slits(tr_fat, dt, E, B, geomT15,
                                      timestep_divider=10)
            fat_beam_list.append(tr_fat)
            if abs(y) < 1e-6:
                break

# %% save zones
dirname = 'output/' + 'B{}_I{}'.format(int(Btor), int(Ipl))
for i_slit in range(n_slits):
    zone = np.empty([0, 3])
    for tr in fat_beam_list:
        zone = np.vstack([zone, tr.ion_zones[i_slit]])
    fname = dirname + '/' + 'E{}'.format(int(Ebeam)) + \
        '_UA2{}'.format(int(UA2)) + '_slit{}.txt'.format(i_slit)
    np.savetxt(fname, zone, fmt='%.4e')

# %% plot results
fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')
# set_axes_param(ax2, 'X (m)', 'Z (m)')
hbplot.set_axes_param(ax3, 'Z (m)', 'Y (m)')
ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
              .format(tr.Ebeam, tr.U[0], Btor, Ipl))

# plot T-15 camera, coils and separatrix on XY plane
geomT15.plot_geom(ax1)
# plot plates
geomT15.plot_plates(ax1, axes='XY')
geomT15.plot_plates(ax3, axes='ZY')

n_slits = geomT15.slits_edges.shape[0]
# set color cycler
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors[:n_slits]
colors = cycle(colors)

# draw slits
r_slits = geomT15.slits_edges
slits_spot = geomT15.slits_spot
ax1.fill(slits_spot[:, 0], slits_spot[:, 1], fill=False)
ax3.fill(slits_spot[:, 2], slits_spot[:, 1], fill=False)
for i in range(n_slits):
    c = next(colors)
    # plot centers
    ax1.plot(r_slits[i, 0, 0], r_slits[i, 0, 1], '*', color=c)
    ax3.plot(r_slits[i, 0, 2], r_slits[i, 0, 1], '*', color=c)
    # plot edges
    ax1.fill(r_slits[i, 1:, 0], r_slits[i, 1:, 1], fill=False)
    ax3.fill(r_slits[i, 1:, 2], r_slits[i, 1:, 1], fill=False)

# plot trajectories
for tr in fat_beam_list:
    # plot primary trajectory
    tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
    tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)
    ax1.plot(tr.RV0[0, 0], tr.RV0[0, 1], 'o',
             color='k', markerfacecolor='white')
    ax3.plot(tr.RV0[0, 2], tr.RV0[0, 1], 'o',
             color='k', markerfacecolor='white')

    # plot secondaries
    for i_slit in range(n_slits):
        c = next(colors)
        for fan_tr in tr.RV_sec_toslits[i_slit]:
            ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
            ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)

    # plot zones
    for i_slit in range(n_slits):
        c = next(colors)
        for fan_tr in tr.RV_sec_toslits[i_slit]:
            ax1.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')
            ax3.plot(fan_tr[0, 2], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')

# %% plot SVs
fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')
# set_axes_param(ax2, 'X (m)', 'Z (m)')
hbplot.set_axes_param(ax3, 'Z (m)', 'Y (m)')
ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
              .format(tr.Ebeam, tr.U[0], Btor, Ipl))

# plot T-15 camera, coils and separatrix on XY plane
geomT15.plot_geom(ax1)
# plot plates
geomT15.plot_plates(ax1, axes='XY')
geomT15.plot_plates(ax3, axes='ZY')
# plot slits
geomT15.plot_slits(ax1, axes='XY')
geomT15.plot_slits(ax3, axes='ZY')

# plot flux surfaces
Psi_vals, x_vals, y_vals, bound_flux = hb.import_Bflux('1MA_sn.txt')
ax1.contour(x_vals, y_vals, Psi_vals, 150)

# plot trajectories
for tr in fat_beam_list:
    # plot primary trajectory
    tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
    tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)

#     # plot secondaries
#     # for i_slit in range(n_slits):
#     #     c = next(colors)
#     #     for fan_tr in tr.RV_sec_toslits[i_slit]:
#     #         ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
#     #         ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)

n_slits = geomT15.slits_edges.shape[0]
# set color cycler
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors[:n_slits]
colors = cycle(colors)

for i_slit in range(1):
    c = next(colors)
    coords = np.empty([0, 3])
    coords_first = np.empty([0, 3])
    coords_last = np.empty([0, 3])
    for tr in fat_beam_list:
        coords_first = np.vstack([coords_first, tr.ion_zones[i_slit][0, 0:3]])
        coords_last = np.vstack([coords_last, tr.ion_zones[i_slit][-1, 0:3]])
        # plot zones of each filament
        ax1.plot(tr.ion_zones[i_slit][:, 0], tr.ion_zones[i_slit][:, 1],
                  'o', color=c, markerfacecolor='white')
        ax3.plot(tr.ion_zones[i_slit][:, 2], tr.ion_zones[i_slit][:, 1],
                  'o', color=c, markerfacecolor='white')
    coords = np.vstack([coords_first, coords_last[::-1]])
    coords = np.vstack([coords, coords[0, :]])
    # ploy in XY plane
    ax1.fill(coords[:, 0], coords[:, 1], '--', color=c)
    ax1.plot(coords[:, 0], coords[:, 1], color='k', lw=0.5)
    # plot in ZY plane
    ax3.fill(coords[:, 2], coords[:, 1], '--', color=c)
    ax3.plot(coords[:, 2], coords[:, 1], color='k', lw=0.5)

