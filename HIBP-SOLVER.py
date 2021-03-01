import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy
import time


# %%
''' MAIN '''

if __name__ == '__main__':
    # timestep [sec]
    dt = 0.7e-7  # 0.4e-7

    # toroidal field on the axis
    Btor = 1.0  # [T]
    Ipl = 1.0  # Plasma current [MA]
    q = 1.60217662e-19  # electron charge [Co]
    m_ion = 204.3833 * 1.6605e-27  # Tl ion mass [kg]

    # initial beam energy range
    dEbeam = 20.
    Ebeam_range = np.arange(240., 240. + dEbeam, dEbeam)  # [keV]

    # A2 plates voltage
    dUA2 = 5.
    UA2_range = np.arange(-20., -20. + dUA2, dUA2)  # [kV]

    # B2 plates voltage
    UB2 = 5.0  # [kV]
    dUB2 = -7.0  # [kV/m]

    # B3 voltages
    UB3 = 10.0  # [kV]
    dUB3 = -15.0  # [kV/m]

    # A3 voltages
    UA3 = 10.0  # [kV]
    dUA3 = -5.0  # [kV/m]

# %% PRIMARY beamline geometry
    geomT15 = hb.Geometry()

    # plasma parameters
    geomT15.R = 1.5  # tokamak major radius [m]
    geomT15.r_plasma = 0.7  # plasma minor radius [m]
    geomT15.elon = 1.8  # plasma elongation

    # alpha and beta angles of the PRIMARY beamline [deg]
    alpha_prim = 20.
    beta_prim = -10.
    gamma_prim = 0.
    geomT15.prim_angles = np.array([alpha_prim, beta_prim, gamma_prim])

    # coordinates of the injection port [m]
    xpatr = 1.5 + 0.726
    ypatr = 1.064
    zpatr = 0.0
    geomT15.r_dict['port'] = np.array([xpatr, ypatr, zpatr])

    # distance from the injection port to the Alpha2 plates
    dist_A2 = 0.35  # [m]
    # distance from the injection port to the Beta2 plates
    dist_B2 = dist_A2 + 0.3  # [m]
    # distance from the injection port to the initial piont of the traj [m]
    dist_r0 = dist_B2 + 0.2

    # coordinates of the center of the ALPHA2 plates
    geomT15.add_coords('A2', 'port', dist_A2, geomT15.prim_angles)

    # coordinates of the center of the BETA2 plates
    geomT15.add_coords('B2', 'port', dist_B2, geomT15.prim_angles)

    # coordinates of the initial point of the trajectory [m]
    geomT15.add_coords('r0', 'port', dist_r0, geomT15.prim_angles)
    r0 = geomT15.r_dict['r0']

# %% AIM position (BEFORE the Secondary beamline) [m]
    xaim = 2.6  # 2.5
    yaim = -0.15  # -0.25
    zaim = 0.0
    r_aim = np.array([xaim, yaim, zaim])
    geomT15.r_dict['aim'] = r_aim

    # angles of aim plane normal [deg]
    alpha_aim = 0.
    beta_aim = 0.
    stop_plane_n = hb.calc_vector(1.0, alpha_aim, beta_aim,
                                  direction=(1, 1, 1))

# %% SECONDARY beamline geometry
    # alpha and beta angles of the SECONDARY beamline [deg]
    alpha_sec = 20.
    beta_sec = 20.
    gamma_sec = -20.
    geomT15.sec_angles = np.array([alpha_sec, beta_sec, gamma_sec])

    # distance from r_aim to the ALPHA3 center
    dist_A3 = 0.3  # 1/2 of plates length
    # distance from r_aim to the BETA3 center
    dist_B3 = dist_A3 + 0.6
    # distance from r_aim the entrance slit of the analyzer
    dist_s = dist_B3 + 0.5

    # coordinates of the center of the ALPHA3 plates
    geomT15.add_coords('A3', 'aim', dist_A3, geomT15.sec_angles)

    # coordinates of the center of the BETA3 plates
    geomT15.add_coords('B3', 'aim', dist_B3, geomT15.sec_angles)

    # Coordinates of the CENTRAL slit
    geomT15.add_coords('slit', 'aim', dist_s, geomT15.sec_angles)

    # Coordinates of the ANALYZER
    geomT15.add_coords('an', 'aim', dist_s, geomT15.sec_angles)

# %% print info
    print('\nShot parameters: Btor = {} T, Ipl = {} MA'. format(Btor, Ipl))
    print('Primary beamline angles: ', geomT15.prim_angles[0:2])
    print('r0 = ', np.round(geomT15.r_dict['r0'], 3))
    print('r_aim = ', r_aim)
    print('r_slit = ', np.round(geomT15.r_dict['slit'], 3))

# %% GEOMETRY
    # chamber entrance and exit coordinates
    geomT15.chamb_ent = [(2.016, 1.069), (2.238, 1.193),
                         (2.211, 0.937), (2.363, 1.04)]
    geomT15.chamb_ext = [(2.39, -0.44), (2.39, -2.0),
                         (2.39, 0.44), (2.39, 0.8)]

    # Toroidal Field coil
    geomT15.coil = np.loadtxt('TFCoil.dat') / 1000  # [m]
    # Poloidal Field coils
    geomT15.pf_coils = hb.import_PFcoils('PFCoils.dat')
    # Camera contour
    geomT15.camera = np.loadtxt('T15_vessel.txt') / 1000
    # Separatrix contour
    geomT15.sep = np.loadtxt('T15_sep.txt') / 1000
    # First wall innner and outer contours
    geomT15.in_fw = np.loadtxt('infw.txt') / 1000  # [m]
    geomT15.out_fw = np.loadtxt('outfw.txt') / 1000  # [m]

# %% Load Electric Field
    ''' Electric field part '''
    # load E for primary beamline
    E_prim, edges_prim = hb.read_E('prim', geomT15)
    geomT15.plates_edges.update(edges_prim)
    print('Primary Beamline loaded')

    # load E for secondary beamline
    try:
        E_sec, edges_sec = hb.read_E('sec', geomT15)
        geomT15.plates_edges.update(edges_sec)
    except FileNotFoundError:
        print('Secondary Beamline NOT FOUND')
        E_sec = []

    E = E_prim + E_sec

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
        print('\nNO Analyzer')

# %% Load Magnetic Field
    ''' Magnetic field part '''
    pf_coils = hb.import_PFcoils('PFCoils.dat')

    PF_dict = hb.import_PFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)
    if 'B' not in locals():
        dirname = 'magfield'
        B = hb.read_B(Btor, Ipl, PF_dict, dirname=dirname)

# %% Optimize Primary Beamline
    print('\n Primary beamline optimization')
    t1 = time.time()
    # define list of trajectores that hit r_aim
    traj_list = []

    for Ebeam in Ebeam_range:
        for UA2 in UA2_range:
            print('\n\nE = {} keV; UA2 = {} kV\n'.format(Ebeam, UA2))
            # list of starting voltages
            U_list = [UA2, UB2, UA3, UB3, Ebeam/(2*G)]

            # create new trajectory object
            tr = hb.Traj(q, m_ion, Ebeam, r0, alpha_prim, beta_prim,
                         U_list, dt)

            tr = hb.optimize_B2(tr, geomT15, UB2, dUB2, E, B, dt,
                                stop_plane_n, eps_xy=1e-3, eps_z=1e-3)

            if tr.IntersectGeometry:
                print('NOT saved, primary intersected geometry')
                continue
            if tr.IsAimXY and tr.IsAimZ:
                traj_list.append(tr)
                print('\n Trajectory saved, UB2={:.2f} kV'.format(tr.U[1]))
                UB2 = tr.U[1]
            else:
                print('NOT saved, sth wrong')

    t2 = time.time()
    print("\n B2 voltage optimized, t = {:.1f} s\n".format(t2-t1))

# %%
    traj_list_passed = copy.deepcopy(traj_list)

# %% Additonal plots

    hbplot.plot_grid(traj_list_passed, geomT15, Btor, Ipl, marker_A2='')
    hbplot.plot_fan(traj_list_passed, geomT15, 240., UA2, Btor, Ipl,
                    plot_analyzer=True, plot_traj=True, plot_all=True)

    # hbplot.plot_scan(traj_list_passed, geomT15, 240., Btor, Ipl)
    # hbplot.plot_scan(traj_list_passed, geomT15, 120., Btor, Ipl)
    # hbplot.plot_sec_angles(traj_list_passed, Btor, Ipl, Ebeam='all')
    # hbplot.plot_fan(traj_list_passed, geomT15, 240., 40., Btor, Ipl)

# %% Optimize Secondary Beamline
    print('\n Secondary beamline optimization')
    t1 = time.time()
    traj_list_a3b3 = []
    for tr in copy.deepcopy(traj_list_passed):
        tr = hb.optimize_A3B3(tr, geomT15, UA3, UB3, dUA3, dUB3, E, B, dt,
                              target='slit', eps_xy=1e-3, eps_z=1e-3)
        if not tr.IntersectGeometrySec:
            traj_list_a3b3.append(tr)
            print('\n Trajectory saved')
            UA3 = tr.U[2]
            UB3 = tr.U[3]
        else:
            print('\n NOT saved')

    t2 = time.time()
    print("\n A3 & B3 voltages optimized, t = {:.1f} s\n".format(t2-t1))

# %%
    hbplot.plot_traj(traj_list_a3b3, geomT15, 240., -8.0, Btor, Ipl,
                     full_primary=False, plot_analyzer=True)
    hbplot.plot_scan(traj_list_a3b3, geomT15, 240., Btor, Ipl)

# %% Pass trajectory to the Analyzer
    print('\n Passing trajectories to Analyzer')
    t1 = time.time()
    traj_list_det = []
    # for tr in copy.deepcopy(traj_list_a3b3):
    #     tr = hb.optimize_A3B3(tr, geomT15, UA3, UB3, dUA3, dUB3, E, B, dt,
    #                           target='det', eps_xy=1e-3, eps_z=1e-3)
    #     if not tr.IntersectGeometrySec:
    #         traj_list_det.append(tr)
    #         print('\n Trajectory saved')
    #         UA3 = tr.U[2]
    #         UB3 = tr.U[3]

    t2 = time.time()
    print("\n Calculation finished, t = {:.1f} s\n".format(t2-t1))

# %%
    hbplot.plot_traj(traj_list_det, geomT15, 240., -8.0, Btor, Ipl,
                     full_primary=False, plot_analyzer=True)
    hbplot.plot_scan(traj_list_det, geomT15, 240., Btor, Ipl)

# %% Save list of trajectories

    # hb.save_traj_list(traj_list_passed, Btor, Ipl, r_aim)
