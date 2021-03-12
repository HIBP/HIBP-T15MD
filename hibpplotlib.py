# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import visvis as vv
import wire
import hibplib as hb
import pylab
import os
from scipy.stats import gaussian_kde
from itertools import cycle
import alphashape

# %%
'''
   ############################################################################
   Functions for plotting Mafnetic field
   ############################################################################
'''


# %% visvis plot 3D
def plot_3d(B, wires, volume_corner1, volume_corner2,
            grid, resolution, cutoff=2):
    '''
    plot absolute values of B in 3d with visvis
    :param B: magnetic field values array (has 3 dimensions) [T]
    :param wires: list of wire objects
    :return: None
    '''

    Babs = np.linalg.norm(B, axis=1)
    Babs[Babs > cutoff] = np.nan

    # draw results
    # prepare axes
    a = vv.gca()
    a.cameraType = '3d'
    a.daspectAuto = False

    vol = Babs.reshape(grid.shape[1:]).T
    vol = vv.Aarray(vol, sampling=(resolution, resolution, resolution),
                    origin=(volume_corner1[2], volume_corner1[1],
                            volume_corner1[0]))
    # set labels
    vv.xlabel('x axis')
    vv.ylabel('y axis')
    vv.zlabel('z axis')

    wire.vv_PlotWires(wires)

    vv.volshow2(vol, renderStyle='mip', cm=vv.CM_JET)
    vv.colorbar()
    app = vv.use()
    app.Run()


# %% matplotlib plot 2D
def plot_2d(B, points, plane='xy', cutoff=2, n_contours=50):
    '''
    make contour plot of B in XZ and XY plane
    :param B: magnetic field values array (has 3 dimensions) [T]
    :param points: coordinates for points for B vectors to start on
    :return: None
    '''
    pf_coils = hb.import_PFcoils('PFCoils.dat')
    # 2d quiver
    # get 2D values from one plane with Y = 0
    fig = plt.figure()
    ax = fig.gca()
    if plane == 'xz':
        # choose y position
        mask = (np.around(points[:, 1], 3) == 0.1)
        B = B[mask]
        Babs = np.linalg.norm(B, axis=1)
        B[Babs > cutoff] = [np.nan, np.nan, np.nan]
        points = points[mask]
#        ax.quiver(points[:, 0], points[:, 2], B[:, 0], B[:, 2], scale=2.0)

        X = np.unique(points[:, 0])
        Z = np.unique(points[:, 2])
        cs = ax.contour(X, Z, Babs.reshape([len(X), len(Z)]).T, n_contours)
        ax.clabel(cs)
        plt.xlabel('x')
        plt.ylabel('z')

    elif plane == 'xy':
        # get coil inner and outer profile
        TF_coil_filename = 'TFCoil.dat'
        # load coil contours from txt
        TF_coil = np.loadtxt(TF_coil_filename) / 1000  # [m]

        # plot toroidal coil
        ax.plot(TF_coil[:, 0], TF_coil[:, 1], '--', color='k')
        ax.plot(TF_coil[:, 2], TF_coil[:, 3], '--', color='k')

        # plot pf coils
        for coil in pf_coils.keys():
            xc = pf_coils[coil][0]
            yc = pf_coils[coil][1]
            dx = pf_coils[coil][2]
            dy = pf_coils[coil][3]
            ax.add_patch(Rectangle((xc-dx/2, yc-dy/2), dx, dy,
                                   linewidth=1, edgecolor='r', facecolor='r'))

        # choose z position
        mask = (np.around(points[:, 2], 3) == 0.0)
        B = B[mask]
        Babs = np.linalg.norm(B, axis=1)
        B[Babs > cutoff] = [np.nan, np.nan, np.nan]
        points = points[mask]
#        ax.quiver(points[:, 0], points[:, 1], B[:, 0], B[:, 1], scale=20.0)

        X = np.unique(points[:, 0])
        Y = np.unique(points[:, 1])
        cs = ax.contour(X, Y, Babs.reshape([len(X), len(Y)]).T, n_contours)
        ax.clabel(cs)

        plt.xlabel('x')
        plt.ylabel('y')

    plt.axis('equal')

    clb = plt.colorbar(cs)
    clb.set_label('V', labelpad=-40, y=1.05, rotation=0)

    plt.show()


# %%
def plot_B_stream(B, volume_corner1, volume_corner2, resolution,
                  grid, color='r', dens=1.0, plot_sep=True):

    fix, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    plot_geometry(ax1, plot_sep=plot_sep)

    x = np.arange(volume_corner1[0], volume_corner2[0], resolution)
    y = np.arange(volume_corner1[1], volume_corner2[1], resolution)
    z = np.arange(volume_corner1[2], volume_corner2[2], resolution)

    Bx = B[:, 0].reshape(grid.shape[1:])
    By = B[:, 1].reshape(grid.shape[1:])
    Bz = B[:, 2].reshape(grid.shape[1:])

    # choose z position
    z_cut = Bx.shape[2]//2  # z position of XY cut
    # choose y position
    y_cut = Bx.shape[1]//2  # z position of XZ cut

    ax1.streamplot(x, y, Bx[:, :, z_cut].swapaxes(0, 1),
                   By[:, :, z_cut].swapaxes(0, 1), color=color, density=dens)
    ax2.streamplot(x, z, Bx[:, y_cut, :].swapaxes(0, 1),
                   Bz[:, y_cut, :].swapaxes(0, 1), color=color, density=dens)


# %% matplotlib plot 3D
def plot_3dm(B, wires, points, cutoff=2):
    '''
    plot 3d quiver of B using matplotlib
    '''
    Babs = np.linalg.norm(B, axis=1)
    B[Babs > cutoff] = [np.nan, np.nan, np.nan]

    fig = plt.figure()
    # 3d quiver
    ax = fig.gca(projection='3d')
    wire.mpl3d_PlotWires(wires, ax)
#    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
#              B[:, 0], B[:, 1], B[:, 2], length=20)
    plt.show()


# %%
'''
   ############################################################################
   Functions for plotting electric field
   ############################################################################
'''


# %%
def plot_contours(X, Y, Z, U, upper_plate_flag, lower_plate_flag,
                  n_contours=30, plates_color='k'):
    '''
    contour plot of potential U
    :param X, Y, Z: mesh ranges in X, Y and Z respectively [m]
    :param U:  plate's U  [V]
    :param n_contours:  number of planes to skip before plotting
    :return: None
    '''

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'Z (m)', 'Y (m)')

    z_cut = U.shape[2]//2
    CS = ax1.contour(X, Y, U[:, :, z_cut].swapaxes(0, 1), n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x_cut = U.shape[0]//2
    ax2.contour(Z, Y, U[x_cut, :, :], n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)

    # add plates
    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

    ax1.plot(x[:, :, z_cut][upper_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][upper_plate_flag[:, :, z_cut]], 'o', color='k')
    ax1.plot(x[:, :, z_cut][lower_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][lower_plate_flag[:, :, z_cut]], 'o', color='k')

    ax2.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='k')
    ax2.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')

    # clb = plt.colorbar(CS)
    # clb.set_label('V', labelpad=-40, y=1.05, rotation=0)


# %%
def plot_contours_zy(X, Y, Z, U, upper_plate_flag, lower_plate_flag,
                     n_contours=30, plates_color='k'):

    fig, ax1 = plt.subplots()
    set_axes_param(ax1, 'Z (m)', 'Y (m)')

    x_cut = U.shape[0]//2
    ax1.contour(Z, Y, U[x_cut, :, :], n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
    ax1.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='r')
    ax1.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')

    # clb = plt.colorbar(CS)
    # clb.set_label('V', labelpad=-40, y=1.05, rotation=0)


# %%
def plot_contours_xz(X, Y, Z, U, upper_plate_flag, lower_plate_flag,
                     n_contours=30, plates_color='k'):

    fig, ax1 = plt.subplots()
    set_axes_param(ax1, 'X (m)', 'Z (m)')

    y_cut = U.shape[1]//2
    ax1.contour(X, Z, U[:, y_cut, :].swapaxes(0, 1), n_contours)
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Z)), max(X)-min(X), max(Z)-min(Z),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
    ax1.plot(x[:, y_cut, :][upper_plate_flag[:, y_cut, :]],
             z[:, y_cut, :][upper_plate_flag[:, y_cut, :]], 'o', color='k')
    ax1.plot(x[:, y_cut, :][lower_plate_flag[:, y_cut, :]],
             z[:, y_cut, :][lower_plate_flag[:, y_cut, :]], 'o', color='k')


# %%
def plot_stream_zy(X, Y, Z, Ex, Ey, Ez, upper_plate_flag, lower_plate_flag,
                   dens=1.0, plates_color='k'):

    fig, ax1 = plt.subplots()
    set_axes_param(ax1, 'Z (m)', 'Y (m)')

    x_cut = Ey.shape[0]//2
    ax1.streamplot(Z, Y, Ez[x_cut, :, :], Ey[x_cut, :, :], density=dens)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')
    ax1.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='r')
    ax1.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')


# %%
def plot_stream(X, Y, Z, Ex, Ey, Ez, upper_plate_flag, lower_plate_flag,
                dens=1.0, plates_color='k'):
    '''
    stream plot of Electric field in xy, xz, zy planes
    :param X, Y, Z: mesh ranges in X, Y and Z respectively [m]
    :param Ex, Ey, Ez: U gradient components [V/m]
    :return: None
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'Z (m)', 'Y (m)')

    z_cut = Ex.shape[2]//2  # z position of XY cut

    ax1.streamplot(X, Y, Ex[:, :, z_cut].swapaxes(0, 1),
                   Ey[:, :, z_cut].swapaxes(0, 1), density=dens)
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x_cut = Ey.shape[0]//2  # x position of ZY cut
    ax2.streamplot(Z, Y, Ez[x_cut, :, :], Ey[x_cut, :, :], density=dens)
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)

    # add plates
    x, y, z = np.meshgrid(X, Y, Z, indexing='ij')

    ax1.plot(x[:, :, z_cut][upper_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][upper_plate_flag[:, :, z_cut]], 'o', color='k')
    ax1.plot(x[:, :, z_cut][lower_plate_flag[:, :, z_cut]],
             y[:, :, z_cut][lower_plate_flag[:, :, z_cut]], 'o', color='k')

    ax2.plot(z[x_cut, :, :][upper_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][upper_plate_flag[x_cut, :, :]], 'o', color='k')
    ax2.plot(z[x_cut, :, :][lower_plate_flag[x_cut, :, :]],
             y[x_cut, :, :][lower_plate_flag[x_cut, :, :]], 'o', color='k')


# %%
def plot_quiver(X, Y, Z, Ex, Ey, Ez):
    '''
    quiver plot of Electric field in xy, xz, zy planes
    :param X, Y, Z: mesh ranges in X, Y and Z respectively [m]
    :param Ex, Ey, Ez: U gradient components [V/m]
    :return: None
    '''
#    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    # fig, ax1 = plt.subplots(nrows=1, ncols=1)

    z_cut = Ex.shape[2]//2 + 20  # z position of XY cut

    ax1.quiver(X, Y, Ex[:, :, z_cut].swapaxes(0, 1),
               Ey[:, :, z_cut].swapaxes(0, 1))
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(X), min(Y)), max(X)-min(X), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax1.add_patch(domain)

    x_cut = Ey.shape[0]//2  # x position of ZY cut
    ax2.quiver(Z, Y, Ez[x_cut, :, :], Ey[x_cut, :, :])
    ax2.set_xlabel('Z (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.axis('equal')
    # add the edge of the domain
    domain = patches.Rectangle((min(Z), min(Y)), max(Z)-min(Z), max(Y)-min(Y),
                               linewidth=2, linestyle='--', edgecolor='k',
                               facecolor='none')
    ax2.add_patch(domain)
#
#    y_cut = Ex.shape[1]//2 # y position of XZ cut
#    ax3.quiver(X, Z, Ex[:, y_cut, :].swapaxes(0, 1),
#                     Ez[:, y_cut, :].swapaxes(0, 1))
#    ax3.set_xlabel('X (m)')
#    ax3.set_ylabel('Z (m)')
#    ax3.grid(True)
#    ax3.axis('equal')
#    # add the edge of the domain
#    domain = patches.Rectangle((min(X), min(Z)), max(X)-min(X), max(Z)-min(Z),
#                               linewidth=2, linestyle='--', edgecolor='k',
#                               facecolor='none')
#    ax3.add_patch(domain)


# %%
def plot_quiver3d(X, Y, Z, Ex, Ey, Ez, UP_rotated, LP_rotated, n_skip=5):
    '''
    3d quiver plot of Electric field
    :param X, Y, Z: mesh ranges in X, Y and Z respectively
    :param Ex, Ey, Ez:  plate's U gradient components
    :param UP_rotated, LP_rotated: upper's and lower's plate angle coordinates
    :param n_skip:  number of planes to skip before plotting
    :return: None
    '''
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plot plates (very scematically)
    ax.plot(UP_rotated[:, 0], UP_rotated[:, 1], UP_rotated[:, 2],
            '-o', color='b')
    ax.plot(LP_rotated[:, 0], LP_rotated[:, 1], LP_rotated[:, 2],
            '-o', color='r')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True)

    x_pos = X.shape[0]//2
    y_pos = Y.shape[1]//2
#    z_pos = Z.shape[2]//2

    skip = (x_pos, slice(None, None, 3*n_skip), slice(None, None, n_skip))
    ax.quiver3D(X[skip], Y[skip], Z[skip],
                Ex[skip],
                Ey[skip],
                Ez[skip], length=0.01, normalize=True)

    skip = (slice(None, None, 3*n_skip), y_pos, slice(None, None, n_skip))
    ax.quiver3D(X[skip], Y[skip], Z[skip],
                Ex[skip],
                Ey[skip],
                Ez[skip], length=0.01, normalize=True)

    ax.axis('equal')


# %%
'''
   ############################################################################
   Functions for plotting trajectories
   ############################################################################
'''


def plot_geometry(ax, TF_coil_filename='TFCoil.dat',
                  camera_data_filename='T15_vessel.txt',
                  separatrix_data_filename='T15_sep.txt',
                  PFCoils_data_filename='PFCoils.dat',
                  major_radius=1.5, plot_sep=True):
    '''
    plot toroidal and poloidal field coils, camera and separatrix in XY plane
    '''
    # load coil contours from txt
    TF_coil = np.loadtxt(TF_coil_filename) / 1000  # [m]

    # plot toroidal coil
    ax.plot(TF_coil[:, 0], TF_coil[:, 1], '--', color='k')
    ax.plot(TF_coil[:, 2], TF_coil[:, 3], '--', color='k')

    # get T-15 camera and plasma contours
    camera = np.loadtxt(camera_data_filename)/1000
    ax.plot(camera[:, 0] + major_radius, camera[:, 1], color='tab:blue')

    # plot first wall
    in_fw = np.loadtxt('infw.txt') / 1000  # [m]
    out_fw = np.loadtxt('outfw.txt') / 1000  # [m]
    ax.plot(in_fw[:, 0], in_fw[:, 1], color='k')
    ax.plot(out_fw[:, 0], out_fw[:, 1], color='k')

    if plot_sep:
        if separatrix_data_filename is not None:
            separatrix = np.loadtxt(separatrix_data_filename)/1000
            ax.plot(separatrix[:, 0] + major_radius, separatrix[:, 1],
                    color='b')  # 'tab:orange')

    if PFCoils_data_filename is not None:
        pf_coils = hb.import_PFcoils(PFCoils_data_filename)

        # plot pf coils
        for coil in pf_coils.keys():
            xc = pf_coils[coil][0]
            yc = pf_coils[coil][1]
            dx = pf_coils[coil][2]
            dy = pf_coils[coil][3]
            ax.add_patch(Rectangle((xc-dx/2, yc-dy/2), dx, dy,
                                   linewidth=1, edgecolor='tab:gray',
                                   facecolor='tab:gray'))


# %%
def set_axes_param(ax, xlabel, ylabel, isequal=True):
    ax.grid(True)
    ax.grid(which='major', color='tab:gray')  # draw primary grid
    ax.minorticks_on()  # make secondary ticks on axes
    # draw secondary grid
    ax.grid(which='minor', color='tab:gray', linestyle=':')
    ax.xaxis.set_tick_params(width=2)  # increase tick size
    ax.yaxis.set_tick_params(width=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isequal:
        ax.axis('equal')


# %%
def plot_traj(traj_list, geom, Ebeam, UA2, Btor, Ipl, full_primary=False,
              plot_analyzer=False, subplots_vertical=False, scale=5):
    '''
    plot primary and secondary trajectories
    :param traj_list: list of trajectories
    :param geom: Geometry object
    :param Ebeam: beam energy [keV]
    :param: UA2: A2 voltage [kV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: None
    '''
    if subplots_vertical:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios':
                                                    [scale, 1]})
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    # plot geometry
    geom.plot_geom(ax1, axes='XY')
    geom.plot_geom(ax2, axes='XZ')
    # plot slits
    if plot_analyzer:
        geom.plot_analyzer(ax1, axes='XY')
        geom.plot_analyzer(ax2, axes='XZ')

    # plot trajectory
    for tr in traj_list:
        if tr.Ebeam == Ebeam and tr.U[0] == UA2:
            # plot primary
            tr.plot_prim(ax1, axes='XY', color='k', full_primary=full_primary)
            tr.plot_prim(ax2, axes='XZ', color='k', full_primary=full_primary)

            ax1.set_title('E={} keV, UA2={} kV, Btor = {} T, Ipl = {} MA'
                          .format(Ebeam, UA2, Btor, Ipl))

            if plot_analyzer and hasattr(tr, 'RV_sec_slit'):
                n_slits = len(tr.RV_sec_slit)
                for i in range(n_slits):
                    for sec_tr in tr.RV_sec_slit[i]:
                        ax1.plot(sec_tr[:, 0], sec_tr[:, 1], color='r')
                        ax2.plot(sec_tr[:, 0], sec_tr[:, 2], color='r')
                        ax1.plot(sec_tr[0, 0], sec_tr[0, 1], 'o', color='r',
                                 markerfacecolor='w')
                        ax2.plot(sec_tr[0, 0], sec_tr[0, 2], 'o', color='r',
                                 markerfacecolor='w')
            else:
                tr.plot_sec(ax1, axes='XY', color='r')
                tr.plot_sec(ax2, axes='XZ', color='r')

            break


# %%
def plot_fan(traj_list, geom, Ebeam, UA2, Btor, Ipl, plot_traj=True,
             plot_last_points=True, plot_analyzer=False, plot_all=False,
             full_primary=True):
    '''
    plot fan of trajectories in xy, xz and zy planes
    :param traj_list: list of trajectories
    :param geom: Geometry object
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: None
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')
    set_axes_param(ax3, 'Z (m)', 'Y (m)')

    # get color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = cycle(colors)
    # get marker cycler
    markers = cycle(('o', 'v', '^', '<', '>', '*', 'D', 'P', 'd'))

    # plot geometry
    geom.plot_geom(ax1, axes='XY')
    geom.plot_geom(ax2, axes='XZ')
    geom.plot_geom(ax3, axes='ZY')
    # plot analyzer
    if plot_analyzer:
        geom.plot_analyzer(ax1, axes='XY')
        geom.plot_analyzer(ax2, axes='XZ')
        geom.plot_analyzer(ax3, axes='ZY')

    ax1.set_title('E={} keV, Btor={} T, Ipl={} MA'
                  .format(Ebeam, Btor, Ipl))

    # invert Z ax
    # ax3.invert_xaxis()
    sec_color = 'r'
    marker = 'o'

    for tr in traj_list:
        if plot_all:
            UA2 = tr.U[0]
            Ebeam_new = tr.Ebeam
            if Ebeam != Ebeam_new:
                Ebeam = Ebeam_new
                sec_color = next(colors)
                marker = next(markers)

        if tr.Ebeam == Ebeam and tr.U[0] == UA2:
            if plot_traj:
                # plot primary
                tr.plot_prim(ax1, axes='XY', color='k',
                             full_primary=full_primary)
                tr.plot_prim(ax2, axes='XZ', color='k',
                             full_primary=full_primary)
                tr.plot_prim(ax3, axes='ZY', color='k',
                             full_primary=full_primary)
                # plot fan of secondaries
                tr.plot_fan(ax1, axes='XY', color=sec_color)
                tr.plot_fan(ax2, axes='XZ', color=sec_color)
                tr.plot_fan(ax3, axes='ZY', color=sec_color)

            if plot_last_points:
                last_points = []
                for i in tr.Fan:
                    last_points.append(i[-1, :])
                last_points = np.array(last_points)

                ax1.plot(last_points[:, 0], last_points[:, 1],
                         '--', c=sec_color)
                ax1.scatter(last_points[:, 0], last_points[:, 1],
                            marker=marker, c='w', edgecolors=sec_color,
                            label='E={:.1f}, UA2={:.1f}'.format(Ebeam, UA2))
                ax2.plot(last_points[:, 0], last_points[:, 2],
                         '--o', c=sec_color, mfc='w', mec=sec_color)
                ax3.plot(last_points[:, 2], last_points[:, 1],
                         '--', c=sec_color)
                ax3.scatter(last_points[:, 2], last_points[:, 1],
                            marker=marker, c='w', edgecolors=sec_color)

            if not plot_all:
                ax1.set_title('E={} keV, UA2={} kV, UB2={:.1f} kV, '
                              'Btor={} T, Ipl={} MA'
                              .format(Ebeam, UA2, tr.U[1], Btor, Ipl))
                break
    ax1.legend()


# %%
def plot_scan(traj_list, geom, Ebeam, Btor, Ipl, full_primary=False,
              plot_analyzer=False, plot_det_line=False,
              subplots_vertical=False, scale=5):
    '''
    plot scan for one beam with particular energy in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param geom: Geometry object
    :param Ebeam: beam energy [keV]
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: None
    '''
    if subplots_vertical:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios':
                                                    [scale, 1]})
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    # plot geometry
    geom.plot_geom(ax1, axes='XY')
    geom.plot_geom(ax2, axes='XZ')
    # plot analyzer
    if plot_analyzer:
        geom.plot_analyzer(ax1, axes='XY')
        geom.plot_analyzer(ax2, axes='XZ')

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    A2list = []
    det_line_x = []
    det_line_y = []

    for tr in traj_list:
        if tr.Ebeam == Ebeam:
            A2list.append(tr.U[0])
            det_line_x.append(tr.RV_sec[0, 0])
            det_line_y.append(tr.RV_sec[0, 1])
            # plot primary
            tr.plot_prim(ax1, axes='XY', color='k', full_primary=full_primary)
            tr.plot_prim(ax2, axes='XZ', color='k', full_primary=full_primary)
            # plot secondary
            if plot_analyzer and hasattr(tr, 'RV_sec_slit'):
                n_slits = len(tr.RV_sec_slit)
                for i in range(n_slits):
                    for sec_tr in tr.RV_sec_slit[i]:
                        ax1.plot(sec_tr[:, 0], sec_tr[:, 1], color=colors[i])
                        ax2.plot(sec_tr[:, 0], sec_tr[:, 2], color=colors[i])
                        ax1.plot(sec_tr[0, 0], sec_tr[0, 1], 'o',
                                 color=colors[i], markerfacecolor='w')
                        ax2.plot(sec_tr[0, 0], sec_tr[0, 2], 'o',
                                 color=colors[i], markerfacecolor='w')
            else:
                tr.plot_sec(ax1, axes='XY', color='r')
                tr.plot_sec(ax2, axes='XZ', color='r')

    if plot_det_line:
        ax1.plot(det_line_x, det_line_y, '--o', color='r')

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))

    ax1.set_title('Ebeam={} keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
                  .format(Ebeam, UA2_min,  UA2_max, Btor, Ipl))


# %%
def plot_grid(traj_list, geom, Btor, Ipl,
              linestyle_A2='--', linestyle_E='-',
              marker_A2='*', marker_E='p',
              traj_color='tab:gray'):
    '''
    plot detector grid in 2 planes: xy, xz
    :param traj_list: list of trajectories
    :param Btor: toroidal magnetic field [T]
    :param Ipl: plasma current [MA]
    :return: None
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')

    # plot geometry
    geom.plot_geom(ax1, axes='XY')
    geom.plot_geom(ax2, axes='XZ')

    # get the list of A2 and Ebeam
    A2list = []
    Elist = []
    for i in range(len(traj_list)):
        A2list.append(traj_list[i].U[0])
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
    ax1.set_title('Eb = [{}, {}] keV, UA2 = [{}, {}] kV,'
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

        ax1.plot(E_grid[:, 0, i_E], E_grid[:, 1, i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(int(Elist[i_E]))+' keV')
        ax2.plot(E_grid[:, 0, i_E], E_grid[:, 2, i_E],
                 linestyle=linestyle_E,
                 marker=marker_E,
                 label=str(int(Elist[i_E]))+' keV')

    # make a grid of constant A2
    for i_A2 in range(0, N_A2, 1):
        k = -1
        for i_tr in range(len(traj_list)):
            if traj_list[i_tr].U[0] == A2list[i_A2]:
                k += 1
                # take the 1-st point of secondary trajectory
                x = traj_list[i_tr].RV_sec[0, 0]
                y = traj_list[i_tr].RV_sec[0, 1]
                z = traj_list[i_tr].RV_sec[0, 2]
                A2_grid[k, :, i_A2] = [x, y, z]

        ax1.plot(A2_grid[:, 0, i_A2], A2_grid[:, 1, i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2], 1))+' kV')
        ax2.plot(A2_grid[:, 0, i_A2], A2_grid[:, 2, i_A2],
                 linestyle=linestyle_A2,
                 marker=marker_A2,
                 label=str(round(A2list[i_A2], 1))+' kV')

    ax1.legend()

#    ax1.set(xlim=(0.9, 4.28), ylim=(-1, 1.5), autoscale_on=False)


# %%
def plot_traj_toslits(tr, geom, Btor, Ipl, plot_fan=True, plot_flux=True):

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    # fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')
    set_axes_param(ax3, 'Z (m)', 'Y (m)')
    ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
                  .format(tr.Ebeam, tr.U[0], Btor, Ipl))

    # plot geometry
    geom.plot_geom(ax1, axes='XY', plot_aim=False)
    geom.plot_geom(ax2, axes='XZ', plot_aim=False)
    geom.plot_geom(ax3, axes='ZY', plot_aim=False)

    # draw slits
    geom.plot_analyzer(ax1, axes='XY')
    geom.plot_analyzer(ax2, axes='XZ')
    geom.plot_analyzer(ax3, axes='ZY')

    n_slits = geom.slits_edges.shape[0]
    # set color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:n_slits]
    colors = cycle(colors)

    # plot primary trajectory
    tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
    tr.plot_prim(ax2, axes='XZ', color='k', full_primary=True)
    tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)

    # plot precise fan of secondaries
    if plot_fan:
        for fan_tr in tr.fan_to_slits:
            ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color='tab:gray')
            ax2.plot(fan_tr[:, 0], fan_tr[:, 2], color='tab:gray')
            ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color='tab:gray')

    # plot secondaries
    for i_slit in range(n_slits):
        c = next(colors)
        for fan_tr in tr.RV_sec_toslits[i_slit]:
            ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
            ax2.plot(fan_tr[:, 0], fan_tr[:, 2], color=c)
            ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)

    # plot zones
    for i_slit in range(n_slits):
        c = next(colors)
        for fan_tr in tr.RV_sec_toslits[i_slit]:
            ax1.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')
            ax3.plot(fan_tr[0, 2], fan_tr[0, 1], 'o', color=c,
                     markerfacecolor='white')

    # plot magnetic flux lines
    if plot_flux:
        Psi_vals, x_vals, y_vals, bound_flux = hb.import_Bflux('1MA_sn.txt')
        ax1.contour(x_vals, y_vals, Psi_vals, 100)


# %%
def plot_fat_beam(fat_beam_list, geom, Btor, Ipl, n_slit='all', scale=3):

    # fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)
    fig, axs = plt.subplots(2, 2, sharex='col', #sharey='row',
                            gridspec_kw={'height_ratios': [scale, 1],
                                         'width_ratios': [scale, 1]})
    ax1, ax2, ax3 = axs[0, 0], axs[1, 0], axs[0, 1]

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    set_axes_param(ax2, 'X (m)', 'Z (m)')
    set_axes_param(ax3, 'Z (m)', 'Y (m)')
    tr = fat_beam_list[0]
    ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
                  .format(tr.Ebeam, tr.U[0], Btor, Ipl))

    # plot geometry
    geom.plot_geom(ax1, axes='XY', plot_aim=False)
    geom.plot_geom(ax2, axes='XZ', plot_aim=False)
    geom.plot_geom(ax3, axes='ZY', plot_aim=False)
    # draw slits
    geom.plot_analyzer(ax1, axes='XY')
    geom.plot_analyzer(ax2, axes='XZ')
    geom.plot_analyzer(ax3, axes='ZY')

    # get number of slits
    n_slits = geom.slits_edges.shape[0]
    # set color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:n_slits]

    if n_slit == 'all':
        slits = range(n_slits)
    else:
        slits = [n_slit]

    # plot trajectories
    for tr in fat_beam_list:
        # plot primary trajectory
        tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
        tr.plot_prim(ax2, axes='XZ', color='k', full_primary=False)
        tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)
        # plot first point
        ax1.plot(tr.RV0[0, 0], tr.RV0[0, 1], 'o',
                 color='k', markerfacecolor='white')
        ax2.plot(tr.RV0[0, 0], tr.RV0[0, 2], 'o',
                 color='k', markerfacecolor='white')
        ax3.plot(tr.RV0[0, 2], tr.RV0[0, 1], 'o',
                 color='k', markerfacecolor='white')

        # plot secondaries
        for i in slits:
            c = colors[i]
            for fan_tr in tr.RV_sec_toslits[i]:
                ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
                ax2.plot(fan_tr[:, 0], fan_tr[:, 2], color=c)
                ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)

                # plot zones
                ax1.plot(fan_tr[0, 0], fan_tr[0, 1], 'o', color=c,
                         markerfacecolor='white')
                ax2.plot(fan_tr[0, 0], fan_tr[0, 2], 'o', color=c,
                         markerfacecolor='white')
                ax3.plot(fan_tr[0, 2], fan_tr[0, 1], 'o', color=c,
                         markerfacecolor='white')


# %%
def plot_svs(fat_beam_list, geom, Btor, Ipl, n_slit='all',
             plot_prim=True, plot_sec=False, plot_zones=True, plot_cut=False,
             plot_flux=False, alpha_xy=10, alpha_zy=20):

    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)

    set_axes_param(ax1, 'X (m)', 'Y (m)')
    # set_axes_param(ax2, 'X (m)', 'Z (m)')
    set_axes_param(ax3, 'Z (m)', 'Y (m)')
    tr = fat_beam_list[0]
    ax1.set_title('E={} keV, UA2={} kV, Btor={} T, Ipl={} MA'
                  .format(tr.Ebeam, tr.U[0], Btor, Ipl))

    # plot geometry
    geom.plot_geom(ax1, axes='XY', plot_aim=False, plot_sep=False)
    geom.plot_geom(ax3, axes='ZY', plot_aim=False)
    geom.plot_analyzer(ax1, axes='XY')
    # geom.plot_analyzer(ax2, axes='XZ')
    geom.plot_analyzer(ax3, axes='ZY')

    # get number of slits
    n_slits = geom.slits_edges.shape[0]
    # set color cycler
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = colors[:n_slits]

    # plot magnetic flux lines
    if plot_flux:
        Psi_vals, x_vals, y_vals, bound_flux = hb.import_Bflux('1MA_sn.txt')
        ax1.contour(x_vals, y_vals, Psi_vals, 100,
                    colors=['k'], linestyles=['-'])

    # plot trajectories
    for tr in fat_beam_list:
        # plot primary trajectory
        if plot_prim:
            tr.plot_prim(ax1, axes='XY', color='k', full_primary=True)
            tr.plot_prim(ax3, axes='ZY', color='k', full_primary=True)
        # plot secondaries
        if plot_sec:
            for i in range(n_slits):
                c = colors[i]
                for fan_tr in tr.RV_sec_toslits[i]:
                    ax1.plot(fan_tr[:, 0], fan_tr[:, 1], color=c)
                    ax3.plot(fan_tr[:, 2], fan_tr[:, 1], color=c)

    if n_slit == 'all':
        slits = reversed(range(n_slits))
    else:
        slits = [n_slit]

    # plot sample volumes
    for i in slits:
        c = colors[i]
        coords = np.empty([0, 3])
        coords_first = np.empty([0, 3])
        coords_last = np.empty([0, 3])
        for tr in fat_beam_list:
            # skip empty arrays
            if tr.ion_zones[i].shape[0] == 0:
                continue
            coords_first = np.vstack([coords_first, tr.ion_zones[i][0, 0:3]])
            coords_last = np.vstack([coords_last, tr.ion_zones[i][-1, 0:3]])
            # plot zones of each filament
            if plot_zones:
                ax1.plot(tr.ion_zones[i][:, 0], tr.ion_zones[i][:, 1],
                         'o', color=c, markerfacecolor='white')
                ax3.plot(tr.ion_zones[i][:, 2], tr.ion_zones[i][:, 1],
                         'o', color=c, markerfacecolor='white')

        if plot_cut:
            coords = np.vstack([coords_first, coords_last[::-1]])
            coords = np.vstack([coords, coords[0, :]])
            # ploy in XY plane
            ax1.fill(coords[:, 0], coords[:, 1], '--', color=c)
            ax1.plot(coords[:, 0], coords[:, 1], color='k', lw=0.5)
            # plot in ZY plane
            ax3.fill(coords[:, 2], coords[:, 1], '--', color=c)
            ax3.plot(coords[:, 2], coords[:, 1], color='k', lw=0.5)
        else:
            coords = np.vstack([coords_first, coords_last])
            # ploy in XY plane
            hull_xy = alphashape.alphashape(coords[:, [0, 1]], alpha_xy)
            hull_pts_xy = hull_xy.exterior.coords.xy
            ax1.fill(hull_pts_xy[0], hull_pts_xy[1], '--', color=c)
            # ax1.plot(hull_pts_xy[0], hull_pts_xy[1], color='k', lw=0.5)
            # plot in ZY plane
            hull_zy = alphashape.alphashape(coords[:, [2, 1]], alpha_zy)
            hull_pts_zy = hull_zy.exterior.coords.xy
            ax3.fill(hull_pts_zy[0], hull_pts_zy[1], '--', color=c)
            # ax3.plot(hull_pts_zy[0], hull_pts_zy[1], color='k', lw=0.5)


# %%
def plot_legend(ax, figure_name):
    """
    plots legend in separate window
    Args:
    :ax - axes to get legnd from
    :figure_name - get figure's name as base for legend's file name
    return figure object
    """
    # create separate figure for legend
    figlegend = plt.figure(num='Legend_for_' + figure_name, figsize=(1, 12))

    # get legend from ax
    figlegend.legend(*ax.get_legend_handles_labels(), loc="center")
    plt.show()

    return figlegend


# %%
def plot_sec_angles(traj_list, Btor, Ipl, Ebeam='all'):

    # plotting params
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    set_axes_param(ax1, 'UA2 (kV)', r'Exit $\alpha$ (grad)')
    set_axes_param(ax2, 'UA2 (kV)', r'Exit $\beta$ (grad)')

    if Ebeam == 'all':
        equal_E_list = np.array([tr.Ebeam for tr in traj_list])
        equal_E_list = np.unique(equal_E_list)
    else:
        equal_E_list = np.array([float(Ebeam)])

    angles_dict = {}

    for Eb in equal_E_list:
        angle_list = []
        for tr in traj_list:
            if tr.Ebeam == Eb:
                # Get vector coords for last point in Secondary traj
                Vx = tr.RV_sec[-1, 3]  # Vx
                Vy = tr.RV_sec[-1, 4]  # Vy
                Vz = tr.RV_sec[-1, 5]  # Vz

                angle_list.append(
                    [tr.U[0], tr.U[1],
                     np.arctan(Vy/np.sqrt(Vx**2 + Vz**2))*180/np.pi,
                     np.arctan(-Vz/Vx)*180/np.pi])

        angles_dict[Eb] = np.array(angle_list)
        ax1.plot(angles_dict[Eb][:, 0], angles_dict[Eb][:, 2],
                 'o', label=str(Eb))
        ax2.plot(angles_dict[Eb][:, 0], angles_dict[Eb][:, 3],
                 'o', label=str(Eb))

    ax1.legend()
    ax2.legend()
    ax1.axis('tight')
    ax2.axis('tight')


# %%
def plot_fan3d(traj_list, geom, Ebeam, UA2, Btor, Ipl,
               azim=0.0, elev=0.0,
               plot_slits=False, plot_all=False):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.grid(True)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    colors = cycle(colors)
    # plot aim dot
    # r_aim = geom.r_dict['aim']
    # ax.plot(r_aim[0], r_aim[1], r_aim[2], '*')

    # # plot slit dot
    # r_aim = geom.r_dict['slit']
    # ax.plot(r_aim[0], r_aim[1], r_aim[2], '*')

    for tr in traj_list:
        if plot_all:
            UA2 = tr.U[0]
            sec_color = next(colors)
        else:
            sec_color = 'r'

        if tr.Ebeam == Ebeam and tr.U[0] == UA2:
            # plot primary
            ax.plot(tr.RV_prim[:, 0], tr.RV_prim[:, 1],
                     tr.RV_prim[:, 2], color='k')

            last_points = []
            for i in tr.Fan:
                ax.plot(i[:, 0], i[:, 1], i[:, 2], color=sec_color)
                last_points.append(i[-1, :])
            last_points = np.array(last_points)
            ax.plot(last_points[:, 0], last_points[:, 1],
                    last_points[:, 2], '--o', color=sec_color)

            ax.set_title('E={} keV, UA2={} kV, UB2={:.1f} kV, '
                          'Btor={} T, Ipl={} MA'
                          .format(Ebeam, UA2, tr.U[1], Btor, Ipl))
            # plt.show()
            if not plot_all:
                break

    for name in geom.plates_edges.keys():
        if name == 'A2':
            color = 'b'
        elif name == 'B2':
            color = 'r'
        else:
            color = 'tab:gray'
        for i in range(2):
            x = list(geom.plates_edges[name][i][:, 0])
            y = list(geom.plates_edges[name][i][:, 1])
            z = list(geom.plates_edges[name][i][:, 2])
            verts = [list(zip(x, y, z))]
            poly = Poly3DCollection(verts, edgecolors='k',
                                    facecolors=color)
            poly.set_alpha(0.5)
            ax.add_collection3d(poly)

    # plot slits
    if plot_slits:
        r_slits = geom.slits_edges
        n_slits = r_slits.shape[0]
        for i in range(n_slits):
            x = list(r_slits[i, 1:, 0])
            y = list(r_slits[i, 1:, 1])
            z = list(r_slits[i, 1:, 2])
            verts = [list(zip(x, y, z))]
            poly = Poly3DCollection(verts, edgecolors='k',
                                    facecolors='b')
            poly.set_alpha(0.5)
            ax.add_collection3d(poly)

    # set position
    ax.view_init(elev=elev, azim=azim)
