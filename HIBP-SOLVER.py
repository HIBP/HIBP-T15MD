import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %%
''' MAIN '''

if __name__ == '__main__':
    # timestep [sec]
    dt = 0.7e-7  # 0.4e-7

    # toroidal field on the axis
    Btor = 1.0  # [T]
    Ipl = 1.0  # Plasma current [MA]
    q = 1.60217662e-19  # electron charge [Co]
    m_Tl = 204.3833 * 1.6605e-27  # Tl ion mass [kg]

    # initial beam energy range
    dEbeam = 20.
    Ebeam_range = np.arange(180., 180. + dEbeam, dEbeam)  # [keV]

    # A2 plates voltage
    dUA2 = 5.
    UA2_range = np.arange(-10., 40. + dUA2, dUA2)  # [kV]

    # B2 plates voltage
    UB2 = 0.0  # [kV]
    dUB2 = 7.0  # [kV/m]

    # B3 voltages
    UB3 = 10.0  # [kV]
    dUB3 = 15.0  # [kV/m]

    # A3 voltages
    UA3 = 10.0  # [kV]
    dUA3 = 5.0  # [kV/m]

    r_plasma = 0.8
    elon = 1.9

# %% PRIMARY beamline geometry
    geomT15 = hb.Geometry()

    # alpha and beta angles of the PRIMARY beamline [deg]
    alpha_prim = 20.
    beta_prim = -10.
    gamma_prim = 0.
    geomT15.prim_angles = np.array([alpha_prim, beta_prim, gamma_prim])

    # coordinates of the injection pipe [m]
    xpatr = 1.5 + 0.726
    ypatr = 1.064
    zpatr = 0.0
    geomT15.r_dict['patr'] = np.array([xpatr, ypatr, zpatr])

    # distance from the injection pipe to the Alpha2 plates
    dist_A2 = 0.35  # [m]
    # distance from the injection pipe to the Beta2 plates
    dist_B2 = dist_A2 + 0.3  # [m]
    # distance from the injection pipe to the initial piont of the traj [m]
    dist_0 = dist_B2 + 0.2

    # convert degrees to radians
    drad = np.pi/180
    # coordinates of the center of the ALPHA2 plates
    xA2 = xpatr + dist_A2*np.cos(alpha_prim*drad) * \
        np.cos(beta_prim*drad)
    yA2 = ypatr + dist_A2*np.sin(alpha_prim*drad)
    zA2 = zpatr - dist_A2*np.cos(alpha_prim*drad) * \
        np.sin(beta_prim*drad)
    rA2 = np.array([xA2, yA2, zA2])
    geomT15.r_dict['A2'] = rA2

    # coordinates of the center of the BETA2 plates
    xB2 = xpatr + dist_B2*np.cos(alpha_prim*drad) * \
        np.cos(beta_prim*drad)
    yB2 = ypatr + dist_B2*np.sin(alpha_prim*drad)
    zB2 = zpatr - dist_B2*np.cos(alpha_prim*drad) * \
        np.sin(beta_prim*drad)
    rB2 = np.array([xB2, yB2, zB2])
    geomT15.r_dict['B2'] = rB2

    # coordinates of the initial point of the trajectory [m]
    x0 = xpatr + dist_0*np.cos(alpha_prim*drad) * \
        np.cos(beta_prim*drad)
    y0 = ypatr + dist_0*np.sin(alpha_prim*drad)
    z0 = zpatr - dist_0*np.cos(alpha_prim*drad) * \
        np.sin(beta_prim*drad)
    r0 = np.array([x0, y0, z0])
    geomT15.r_dict['r0'] = r0

# %% AIM position (BEFORE the Secondary beamline) [m]
    xaim = 2.5
    yaim = 0.0  # -0.25
    zaim = 0.0
    r_aim = np.array([xaim, yaim, zaim])
    geomT15.r_dict['aim'] = r_aim

    # angles of aim plane normal [deg]
    alpha_aim = 0.
    beta_aim = 0.
    stop_plane_n = np.array([np.cos(alpha_aim*drad)*np.cos(beta_aim*drad),
                             np.sin(alpha_aim*drad),
                             np.cos(alpha_aim*drad)*np.sin(beta_aim*drad)])
    stop_plane_n = stop_plane_n/np.linalg.norm(stop_plane_n)

# %% SECONDARY beamline geometry
    # alpha and beta angles of the SECONDARY beamline [deg]
    alpha_sec = 30.
    beta_sec = 20.
    gamma_sec = 0.
    geomT15.sec_angles = np.array([alpha_sec, beta_sec, gamma_sec])

    # distance from r_aim to the ALPHA3 center
    dist_A3 = 0.3  # 1/2 of plates length
    # distance from r_aim to the BETA3 center
    dist_B3 = dist_A3 + 0.6
    # distance from r_aim the entrance slit of the analyzer
    dist_s = dist_B3 + 0.5

    # coordinates of the center of the ALPHA3 plates
    xA3 = xaim + dist_A3*np.cos(alpha_sec*drad) * \
        np.cos(beta_sec*drad)
    yA3 = yaim + dist_A3*np.sin(alpha_sec*drad)
    zA3 = zaim - dist_A3*np.cos(alpha_sec*drad) * \
        np.sin(beta_sec*drad)
    rA3 = np.array([xA3, yA3, zA3])
    geomT15.r_dict['A3'] = rA3

    # coordinates of the center of the BETA3 plates
    xB3 = xaim + dist_B3*np.cos(alpha_sec*drad) * \
        np.cos(beta_sec*drad)
    yB3 = yaim + dist_B3*np.sin(alpha_sec*drad)
    zB3 = zaim - dist_B3*np.cos(alpha_sec*drad) * \
        np.sin(beta_sec*drad)
    rB3 = np.array([xB3, yB3, zB3])
    geomT15.r_dict['B3'] = rB3

    # Coordinates of the CENTRAL slit
    xs = xaim + dist_s*np.cos(alpha_sec*drad) * \
        np.cos(beta_sec*drad)
    ys = yaim + dist_s*np.sin(alpha_sec*drad)
    zs = zaim - dist_s*np.cos(alpha_sec*drad) * \
        np.sin(beta_sec*drad)
    rs = np.array([xs, ys, zs])
    geomT15.r_dict['slit'] = rs

# %% print info
    print('\nShot parameters: Btor = {} T, Ipl = {} MA'. format(Btor, Ipl))
    print('Primary beamline angles: ', geomT15.prim_angles[0:2])
    print('r0 = ', np.round(r0, 3))
    print('r_aim = ', r_aim)
    print('r_slit = ', np.round(rs, 3))

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

    E_sec, edges_sec = hb.read_E('sec', geomT15)
    geomT15.plates_edges.update(edges_sec)

    E = E_prim + E_sec
    # E = E_prim

# %% Load Magnetic Field
    ''' Magnetic field part '''
    pf_coils = hb.import_PFcoils('PFCoils.dat')

    PF_dict = hb.import_PFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)
    if 'B' not in locals():
        dirname = 'magfield'
        B = hb.read_B(Btor, Ipl, PF_dict, dirname)

# %% Optimize Primary Beamline
    print('\n Primary beamline optimization')
    # define list of trajectores that hit r_aim
    traj_list = []

    for Ebeam in Ebeam_range:
        for UA2 in UA2_range:
            print('\n\nE = {} keV; UA2 = {} kV\n'.format(Ebeam, UA2))
            # list of starting voltages
            U_list = [UA2, UB2, UA3, UB3]

            # create new trajectory object
            tr = hb.Traj(q, m_Tl, Ebeam, r0, alpha_prim, beta_prim,
                         U_list, dt)

            tr = hb.optimize_B2(tr, r_aim, geomT15, UB2, dUB2, E, B, dt,
                                stop_plane_n, r_plasma, elon,
                                eps_xy=1e-3, eps_z=1e-3)

            if tr.IntersectGeometry:
                print('NOT saved, primary intersected geometry')
                continue
            if tr.IsAimXY and tr.IsAimZ:
                traj_list.append(tr)
                print('\n Trajectory saved, UB2={:.2f} kV'.format(tr.U[1]))

# %%
    traj_list_passed = copy.deepcopy(traj_list)

# %% Additonal plots

    hbplot.plot_grid(traj_list_passed, geomT15, Btor, Ipl, marker_E='')
    hbplot.plot_fan(traj_list_passed, geomT15, 240., UA2, Btor, Ipl,
                    plot_slits=True, plot_traj=True, plot_all=True)

    # hbplot.plot_scan(traj_list_passed, geomT15, 240., Btor, Ipl)
    # hbplot.plot_scan(traj_list_passed, geomT15, 120., Btor, Ipl)
    # hbplot.plot_sec_angles(traj_list_passed, Btor, Ipl, Ebeam='all')
    # hbplot.plot_fan(traj_list_passed, geomT15, 240., 40., Btor, Ipl)

# %% Optimize Secondary Beamline
    print('\n Secondary beamline optimization')
    traj_list_oct = []
    for tr in copy.deepcopy(traj_list_passed):
        tr = hb.optimize_A3B3(tr, rs, geomT15,
                              UA3, UB3, dUA3, dUB3, E, B, dt,
                              eps_xy=1e-3, eps_z=1e-3)
        if not tr.IntersectGeometrySec:
            traj_list_oct.append(tr)

# %%
    hbplot.plot_traj(traj_list_oct, geomT15, 240., 40., Btor, Ipl)
    hbplot.plot_scan(traj_list_oct, geomT15, 240., Btor, Ipl)

# %% Save list of trajectories

    # hb.save_traj_list(traj_list_passed, Btor, Ipl, r_aim)
