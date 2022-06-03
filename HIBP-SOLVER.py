# -*- coding: utf-8 -*-
'''
T-15MD tokamak, HIBP

Program calculates trajectories and selects voltages on
primary beamline (B2 plates) and secondary beamline (A3, B3, A4 plates)
'''

import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import define_geometry as defgeom
import copy
import time
import sys

# %% set up main parameters
# choose analyzer number
analyzer = 1

# toroidal field on the axis
Btor = 1.0  # [T]
Ipl = 1.0  # Plasma current [MA]
print('\nShot parameters: Btor = {} T, Ipl = {} MA'. format(Btor, Ipl))

# timestep [sec]
dt = 0.4e-7  # 0.7e-7

# probing ion charge and mass
q = 1.602176634e-19  # electron charge [Co]
m_ion = 204.3833 * 1.6605e-27  # Tl ion mass [kg]

# beam energy
Emin, Emax, dEbeam = 240., 240., 20.

# set flags
optimizeB2 = True
optimizeA3B3 = True
calculate_zones = False
pass2AN = True
save_radref = False

# UA2 voltages
UA2min, UA2max, dUA2 = -3., 33., 3.  # -3, 33., 3.  # -3., 30., 3.
NA2_points = 10

# B2 plates voltage
UB2, dUB2 = -3.0, 10.  # [kV], [kV/m]

# B3 voltages
UB3, dUB3 = 0.0, 10  # [kV], [kV/m]

# A3 voltages
UA3, dUA3 = 0.0, 7.0  # [kV], [kV/m]
if analyzer == 2:
    dUA3 = -dUA3

# A4 voltages
UA4, dUA4 = 0.0, 2.0  # [kV], [kV/m]

# %% Define Geometry
geomT15 = defgeom.define_geometry(analyzer=analyzer)
r0 = geomT15.r_dict['r0']  # trajectory starting point

# angles of aim plane normal [deg]
alpha_aim = 0.
beta_aim = 0.
stop_plane_n = hb.calc_vector(1.0, alpha_aim, beta_aim,
                              direction=(1, 1, 1))

# %% Load Electric Field
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
    # add diafragm for A3 plates to Geometry
    hb.add_diafragm(geomT15, 'A3', 'A3d', diaf_width=0.05)
    hb.add_diafragm(geomT15, 'A4', 'A4d', diaf_width=0.05)
    print('\n Secondary Beamline loaded')
except FileNotFoundError:
    print('\n Secondary Beamline NOT FOUND')

# %% Analyzer parameters
if 'an' in geomT15.plates_dict.keys():
    # Analyzer G
    G = geomT15.plates_dict['an'].G
    # add detector coords to dictionary
    edges = geomT15.plates_dict['an'].det_edges
    geomT15.r_dict['det'] = edges[edges.shape[0]//2][0]
else:
    G = 1.
    print('\nNO Analyzer')

# %% Load Magnetic Field
pf_coils = hb.import_PFcoils('PFCoils.dat')
PF_dict = hb.import_PFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)
if 'B' not in locals():
    dirname = 'magfield'
    B = hb.read_B(Btor, Ipl, PF_dict, dirname=dirname)

# %% Optimize Primary Beamline
# define list of trajectories that hit r_aim
traj_list_B2 = []
# initial beam energy range
Ebeam_range = np.arange(Emin, Emax + dEbeam, dEbeam)  # [keV]

for Ebeam in Ebeam_range:
    t1 = time.time()
    shot = ''
    input_fname = ''
    print('\n>>INPUT FILE: ', input_fname)
    if input_fname != '':
        exp_voltages = np.loadtxt(input_fname)
        indexes = np.linspace(1, exp_voltages.shape[0]-1,
                              NA2_points, dtype=int)
    if optimizeB2:
        optimizeA3B3 = True
        target = 'aim'
        # A2 plates voltage
        UA2_range = np.arange(UA2min, UA2max + dUA2, dUA2)
        # UA2_range = np.linspace(UA2min, UA2max, NA2_points)  # [kV]
        eps_xy, eps_z = 1e-3, 1e-3
    else:
        target = 'aim'
        UA2_range = exp_voltages[indexes, 1]
        UB2_range = exp_voltages[indexes, 2]
        eps_xy, eps_z = 1e-3, 1.
    if not optimizeA3B3:
        target = 'aim'
        UA3_range = exp_voltages[indexes, 3]
        UB3_range = exp_voltages[indexes, 4]
        eps_xy, eps_z = 1e-3, 1.
    if optimizeB2:
        print('\n Primary beamline optimization')
    else:
        print('\n Calculating primary beamline')

    # UA2 loop
    for i in range(UA2_range.shape[0]):
        UA2 = UA2_range[i]
        if not optimizeB2:
            UB2 = UB2_range[i]
        if not optimizeA3B3:
            UA3, UB3 = UA3_range[i], UB3_range[i]
        print('\n\nE = {} keV; UA2 = {:.2f} kV\n'.format(Ebeam, UA2))
        # dict of starting voltages
        U_dict = {'A2': UA2, 'B2': UB2,
                  'A3': UA3, 'B3': UB3, 'A4': UA4, 'an': Ebeam/(2*G)}
        # create new trajectory object
        tr = hb.Traj(q, m_ion, Ebeam, r0, geomT15.angles_dict['r0'][0],
                     geomT15.angles_dict['r0'][1], U_dict, dt)
        # optimize B2 voltage
        tr = hb.optimize_B2(tr, geomT15, UB2, dUB2, E, B, dt, stop_plane_n,
                            target, optimizeB2, eps_xy=eps_xy, eps_z=eps_z)
        # check geometry intersection
        if True in tr.IntersectGeometry.values():
            print('NOT saved, primary intersected geometry')
            continue
        if True in tr.IntersectGeometrySec.values():
            print('NOT saved, secondary intersected geometry')
            continue
        # if no intersections, upldate UB2 values
        UB2 = tr.U['B2']
        # check aim
        if tr.IsAimXY and tr.IsAimZ:
            traj_list_B2.append(tr)
            print('\n Trajectory saved, UB2={:.2f} kV'.format(tr.U['B2']))
        else:
            print('NOT saved, sth is wrong')
        # traj_list_B2.append(tr)

t2 = time.time()
if optimizeB2:
    print('\n B2 voltage optimized, t = {:.1f} s\n'.format(t2-t1))
else:
    print('\n Trajectories to r_aim calculated, t = {:.1f} s\n'.format(t2-t1))

# %%
traj_list_passed = copy.deepcopy(traj_list_B2)

# %% Save traj list
# hb.save_traj_list(traj_list_passed, Btor, Ipl, geomT15.r_dict[target])
# sys.exit()

# %% Additional plots
hbplot.plot_grid(traj_list_passed, geomT15, Btor, Ipl,
                 onlyE=False, marker_A2='')
# hbplot.plot_fan(traj_list_passed, geomT15, Ebeam, UA2, Btor, Ipl,
#                 plot_analyzer=False, plot_traj=True, plot_all=False)

hbplot.plot_scan(traj_list_passed, geomT15, Ebeam, Btor, Ipl,
                 full_primary=False, plot_analyzer=True,
                 plot_det_line=True, subplots_vertical=True, scale=4)
# hbplot.plot_sec_angles(traj_list_passed, Btor, Ipl, Ebeam='all')
# hbplot.plot_fan(traj_list_passed, geomT15, 240., 40., Btor, Ipl)

# %% Optimize Secondary Beamline
t1 = time.time()
# define list of trajectories that hit slit
traj_list_a3b3 = []
if optimizeA3B3:
    print('\n Secondary beamline optimization')
    for tr in copy.deepcopy(traj_list_passed):
        tr, vltg_fail = hb.optimize_A3B3(tr, geomT15, UA3, UB3, dUA3, dUB3,
                                         E, B, dt, target='slit',
                                         UA3_max=40., UB3_max=40.,
                                         eps_xy=1e-3, eps_z=1e-3)
        # check geometry intersection and voltage failure
        if not (True in tr.IntersectGeometrySec.values()) and not vltg_fail:
            traj_list_a3b3.append(tr)
            print('\n Trajectory saved')
            # UA3 = tr.U['A3']
            # UB3 = tr.U['B3']
        else:
            print('\n NOT saved')
    t2 = time.time()
    print('\n A3 & B3 voltages optimized, t = {:.1f} s\n'.format(t2-t1))
else:
    print('\n Calculating secondary beamline')
    for tr in copy.deepcopy(traj_list_passed):
        print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A2']))
        RV0 = np.array([tr.RV_sec[0]])
        tr.pass_sec(RV0, geomT15.r_dict['slit'], E, B, geomT15,
                    stop_plane_n=geomT15.plates_dict['an'].det_plane_n,
                    tmax=9e-5, eps_xy=eps_xy, eps_z=eps_z)
        traj_list_a3b3.append(tr)
    t2 = time.time()
    print('\n Secondary beamline calculated, t = {:.1f} s\n'.format(t2-t1))

# %% Calculate ionization zones
if calculate_zones:
    t1 = time.time()
    slits = [2]
    traj_list_zones = []
    print('\n Ionization zones calculation')
    for tr in copy.deepcopy(traj_list_a3b3):
        print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A2']))
        tr = hb.calc_zones(tr, dt, E, B, geomT15, slits=slits,
                           timestep_divider=6,
                           eps_xy=1e-4, eps_z=1, dt_min=1e-12,
                           no_intersect=True, no_out_of_bounds=True)
        traj_list_zones.append(tr)
        print('\n Trajectory saved')
    t2 = time.time()
    print('\n Ionization zones calculated, t = {:.1f} s\n'.format(t2-t1))

    hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl,
                             slits=slits, plot_fan=False)

# %% Pass to ANALYZER
if pass2AN:
    print('\n Passing to ANALYZER {}'.format(analyzer))
    # define list of trajectories that hit detector
    traj_list_an = []
    for tr in copy.deepcopy(traj_list_a3b3):
        print('\nEb = {}, UA2 = {:.2f}'.format(tr.Ebeam, tr.U['A2']))
        RV0 = np.array([tr.RV_sec[0]])
        # pass secondary trajectory to detector
        tr.pass_sec(RV0, geomT15.r_dict['det'], E, B, geomT15,
                    stop_plane_n=geomT15.plates_dict['an'].det_plane_n,
                    tmax=9e-5, eps_xy=eps_xy, eps_z=eps_z)
        traj_list_an.append(tr)

# %% Additional plots
# hbplot.plot_grid_a3b3(traj_list_a3b3, geomT15, Btor, Ipl,
#                       linestyle_A2='--', linestyle_E='-',
#                       marker_E='p')
# hbplot.plot_traj(traj_list_a3b3, geomT15, Ebeam, 0.0, Btor, Ipl,
#                   full_primary=False, plot_analyzer=True,
#                   subplots_vertical=True, scale=3.5)
hbplot.plot_scan(traj_list_a3b3, geomT15, Ebeam, Btor, Ipl,
                 full_primary=False, plot_analyzer=True,
                 plot_det_line=False, subplots_vertical=True, scale=5)

hbplot.plot_scan(traj_list_an, geomT15, Ebeam, Btor, Ipl,
                 full_primary=False, plot_analyzer=True,
                 plot_det_line=False, subplots_vertical=True, scale=5)

# %% Pass trajectory to the Analyzer
# print('\n Optimizing entrance angle to Analyzer with A4')
# t1 = time.time()
# traj_list_a4 = []
# for tr in copy.deepcopy(traj_list_a3b3):
#     tr = hb.optimize_A4(tr, geomT15, UA4, dUA4,
#                         E, B, dt, eps_alpha=0.05)
#     if not tr.IntersectGeometrySec:
#         traj_list_a4.append(tr)
#         print('\n Trajectory saved')
#         UA4 = tr.U['A4']

# t2 = time.time()
# print("\n Calculation finished, t = {:.1f} s\n".format(t2-t1))

# %%
# hbplot.plot_traj(traj_list_a4, geomT15, 240., 0.0, Btor, Ipl,
#                   full_primary=False, plot_analyzer=True)
# hbplot.plot_scan(traj_list_a4, geomT15, 240., Btor, Ipl,
#                   full_primary=False, plot_analyzer=False,
#                   plot_det_line=False, subplots_vertical=True, scale=5)

# %% Save list of trajectories
# hb.save_traj_list(traj_list_passed, Btor, Ipl, geomT15.r_dict[target])
