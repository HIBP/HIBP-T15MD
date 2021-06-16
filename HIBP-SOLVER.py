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

# %% set up main parameters
input_fname = ''
NA2_points = 10
if input_fname != '':
    exp_voltages = np.loadtxt(input_fname)
    indexes = np.linspace(1, exp_voltages.shape[0]-1, NA2_points, dtype=int)
# UA2 voltages
UA2min, UA2max, dUA2 = -3., 30., 3.

optimizeB2 = True
optimizeA3B3 = False

if optimizeB2:
    optimizeA3B3 = True
    target = 'aim'
    # A2 plates voltage
    dUA2 = 2.
    # UA2_range = np.linspace(UA2min, UA2max, NA2_points)  # [kV]
    UA2_range = np.arange(UA2min, UA2max, dUA2)  # [kV]
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

# timestep [sec]
dt = 0.4e-7  # 0.7e-7

# toroidal field on the axis
Btor = 1.0  # [T]
Ipl = 1.0  # Plasma current [MA]
print('\nShot parameters: Btor = {} T, Ipl = {} MA'. format(Btor, Ipl))

# probing ion charge and mass
q = 1.60217662e-19  # electron charge [Co]
m_ion = 204.3833 * 1.6605e-27  # Tl ion mass [kg]

# choose analyzer number
analyzer = 1

# initial beam energy range
dEbeam = 20.
Ebeam_range = np.arange(160., 160. + dEbeam, dEbeam)  # [keV]

# B2 plates voltage
UB2 = 0.0  # [kV]
dUB2 = 10.0  # [kV/m]

# B3 voltages
UB3 = 0.0  # [kV]
dUB3 = 10.  # 20.0  # [kV/m]

# A3 voltages
UA3 = 0.0  # [kV]
dUA3 = 7.0  # [kV/m]
if analyzer == 2: dUA3 = -dUA3

# A4 voltages
UA4 = 0.0  # [kV]
dUA4 = 2.0  # [kV/m]

# %% Define Geometry
geomT15 = defgeom.define_geometry(analyzer=analyzer)
r0 = geomT15.r_dict['r0']  # trajectory starting point

# angles of aim plane normal [deg]
alpha_aim = 0.
beta_aim = 0.
stop_plane_n = hb.calc_vector(1.0, alpha_aim, beta_aim,
                                direction=(1, 1, 1))

# %% Load Magnetic Field
pf_coils = hb.import_PFcoils('PFCoils.dat')

PF_dict = hb.import_PFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)
if 'B' not in locals():
    dirname = 'magfield'
    B = hb.read_B(Btor, Ipl, PF_dict, dirname=dirname)

# %% Load Electric Field
E = {}
# load E for primary beamline
try:
    E_prim, edges_prim = hb.read_E('prim', geomT15)
    geomT15.plates_edges.update(edges_prim)
    print('\n Primary Beamline loaded')
except FileNotFoundError:
    print('\n Primary Beamline NOT FOUND')
    E_prim = {}

# load E for secondary beamline
try:
    E_sec, edges_sec = hb.read_E('sec', geomT15)
    geomT15.plates_edges.update(edges_sec)
    print('\n Secondary Beamline loaded')
except FileNotFoundError:
    print('\n Secondary Beamline NOT FOUND')
    E_sec = {}

E.update(E_prim)
E.update(E_sec)

# %% Analyzer parameters
if geomT15.an_params.shape[0] > 0:
    # Analyzer G
    G = geomT15.an_params[3]

    n_slits, slit_dist, slit_w = geomT15.an_params[:3]

    # add slits to Geometry
    geomT15.add_slits(n_slits=n_slits, slit_dist=slit_dist, slit_w=slit_w,
                        slit_l=0.1)

    # define detector
    geomT15.add_detector(n_det=n_slits, det_dist=slit_dist,
                            det_w=slit_dist, det_l=0.1)
    print('\nAnalyzer with {} slits added to Geometry'.format(n_slits))
    print('G = {}\n'.format(G))
else:
    G = 1.
    print('\nNO Analyzer')

# %% Optimize Primary Beamline
if optimizeB2:
    print('\n Primary beamline optimization')
else:
    print('\n Calculating primary beamline')
t1 = time.time()
# define list of trajectories that hit r_aim
traj_list_B2 = []
# main loop
for Ebeam in Ebeam_range:
    for i in range(UA2_range.shape[0]):
        UA2 = UA2_range[i]
        if not optimizeB2: UB2 = UB2_range[i]
        if not optimizeA3B3: UA3, UB3 = UA3_range[i], UB3_range[i]
        print('\n\nE = {} keV; UA2 = {} kV\n'.format(Ebeam, UA2))
        # dict of starting voltages
        U_dict= {'A2':UA2, 'B2':UB2,
                 'A3':UA3, 'B3':UB3, 'A4':UA4, 'an':Ebeam/(2*G)}

        # create new trajectory object
        tr = hb.Traj(q, m_ion, Ebeam, r0,
                        geomT15.angles['r0'][0], geomT15.angles['r0'][1],
                        U_dict, dt)

        tr = hb.optimize_B2(tr, geomT15, UB2, dUB2, E, B, dt, stop_plane_n,
                            'aim', optimizeB2, eps_xy=eps_xy, eps_z=eps_z)
        UB2 = tr.U['B2']

        if True in tr.IntersectGeometry.values():
            print('NOT saved, primary intersected geometry')
            continue
        if True in tr.IntersectGeometrySec.values():
            print('NOT saved, secondary intersected geometry')
            continue
        if tr.IsAimXY and tr.IsAimZ:
            traj_list_B2.append(tr)
            print('\n Trajectory saved, UB2={:.2f} kV'.format(tr.U['B2']))
        else:
            print('NOT saved, sth wrong')

t2 = time.time()
if optimizeB2:
    print('\n B2 voltage optimized, t = {:.1f} s\n'.format(t2-t1))
else:
    print('\n Trajectories to r_aim calculated, t = {:.1f} s\n'.format(t2-t1))

# %%
traj_list_passed = copy.deepcopy(traj_list_B2)

# %% Save traj list

# hb.save_traj_list(traj_list_passed, Btor, Ipl, geomT15.r_dict['aim'])

# %% Additional plots

hbplot.plot_grid(traj_list_passed, geomT15, Btor, Ipl, marker_A2='')
# hbplot.plot_fan(traj_list_passed, geomT15, 240., UA2, Btor, Ipl,
#                 plot_analyzer=False, plot_traj=True, plot_all=False)

hbplot.plot_scan(traj_list_passed, geomT15, Ebeam, Btor, Ipl)
# hbplot.plot_scan(traj_list_passed, geomT15, 120., Btor, Ipl)
# hbplot.plot_sec_angles(traj_list_passed, Btor, Ipl, Ebeam='all')
# hbplot.plot_fan(traj_list_passed, geomT15, 240., 40., Btor, Ipl)

# %% Optimize Secondary Beamline
t1 = time.time()
if optimizeA3B3:
    print('\n Secondary beamline optimization')
    traj_list_a3b3 = []
    for tr in copy.deepcopy(traj_list_passed):
        tr, vltg_fail = hb.optimize_A3B3(tr, geomT15, UA3, UB3, dUA3, dUB3,
                                            E, B, dt, target='slit',
                                            UA3_max=40., UB3_max=40.,
                                            eps_xy=1e-3, eps_z=1e-3)
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
    traj_list_a3b3 = []
    for tr in copy.deepcopy(traj_list_passed):
        print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U['A2']))
        RV0 = np.array([tr.RV_sec[0]])
        tr.pass_sec(RV0, geomT15.r_dict['slit'], E, B, geomT15,
                    # stop_plane_n=stop_plane_n,
                    tmax=9e-5, eps_xy=eps_xy, eps_z=eps_z)
        traj_list_a3b3.append(tr)
    t2 = time.time()
    print('\n Secondary beamline calculated, t = {:.1f} s\n'.format(t2-t1))

# %% Additional plots
hbplot.plot_grid_a3b3(traj_list_a3b3, geomT15, Btor, Ipl,
                      linestyle_A2='--', linestyle_E='-',
                      marker_E='p')
hbplot.plot_traj(traj_list_a3b3, geomT15, 240., 0.0, Btor, Ipl,
                 full_primary=False, plot_analyzer=True,
                 subplots_vertical=True, scale=3.5)
hbplot.plot_scan(traj_list_a3b3, geomT15, 240., Btor, Ipl,
                 full_primary=False, plot_analyzer=False,
                 plot_det_line=False, subplots_vertical=True, scale=5)

# %% Pass trajectory to the Analyzer
#     print('\n Optimizing entrance angle to Analyzer with A4')
#     t1 = time.time()
#     traj_list_a4 = []
#     for tr in copy.deepcopy(traj_list_a3b3):
#         tr = hb.optimize_A4(tr, geomT15, UA4, dUA4,
#                             E, B, dt, eps_alpha=0.05)
#         if not tr.IntersectGeometrySec:
#             traj_list_a4.append(tr)
#             print('\n Trajectory saved')
#             UA4 = tr.U['A4']

#     t2 = time.time()
#     print("\n Calculation finished, t = {:.1f} s\n".format(t2-t1))

# %%
# hbplot.plot_traj(traj_list_a4, geomT15, 240., 0.0, Btor, Ipl,
#                   full_primary=False, plot_analyzer=True)
# hbplot.plot_scan(traj_list_a4, geomT15, 240., Btor, Ipl,
#                   full_primary=False, plot_analyzer=False,
#                   plot_det_line=False, subplots_vertical=True, scale=5)

# %% Save list of trajectories

# hb.save_traj_list(traj_list_passed, Btor, Ipl, geomT15.r_dict['aim'])
