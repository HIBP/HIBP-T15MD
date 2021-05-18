import numpy as np
import hibplib as hb


def define_geometry(Btor, Ipl):
    geom = hb.Geometry()

    # plasma parameters
    geom.R = 1.5  # tokamak major radius [m]
    geom.r_plasma = 0.7  # plasma minor radius [m]
    geom.elon = 1.8  # plasma elongation

    # PRIMARY beamline geometry
    # alpha and beta angles of the PRIMARY beamline [deg]
    alpha_prim = 30.  # 20.
    beta_prim = -10.
    gamma_prim = 0.
    geom.prim_angles = np.array([alpha_prim, beta_prim, gamma_prim])

    # coordinates of the injection port [m]
    xpatr = 1.5 + 0.726
    ypatr = 1.064
    zpatr = 0.0
    geom.r_dict['port'] = np.array([xpatr, ypatr, zpatr])

    # distance from the injection port to the Alpha2 plates
    dist_A2 = 0.4  # [m]
    # distance from the injection port to the Beta2 plates
    dist_B2 = dist_A2 + 0.3  # [m]
    # distance from the injection port to the initial piont of the traj [m]
    dist_r0 = dist_B2 + 0.2

    # coordinates of the center of the ALPHA2 plates
    geom.add_coords('A2', 'port', dist_A2, geom.prim_angles)

    # coordinates of the center of the BETA2 plates
    geom.add_coords('B2', 'port', dist_B2, geom.prim_angles)

    # coordinates of the initial point of the trajectory [m]
    geom.add_coords('r0', 'port', dist_r0, geom.prim_angles)

    # AIM position (BEFORE the Secondary beamline) [m]
    xaim = 2.6  # 2.5
    yaim = -0.25
    zaim = 0.0
    r_aim = np.array([xaim, yaim, zaim])
    geom.r_dict['aim'] = r_aim

    # SECONDARY beamline geometry
    # alpha and beta angles of the SECONDARY beamline [deg]
    alpha_sec = 15.
    beta_sec = 20.
    gamma_sec = -20.
    geom.sec_angles = np.array([alpha_sec, beta_sec, gamma_sec])

    # distance from r_aim to the ALPHA3 center
    dist_A3 = 0.3  # 1/2 of plates length
    # distance from r_aim to the BETA3 center
    dist_B3 = dist_A3 + 0.6
    # from r_aim to A4
    dist_A4 = dist_B3 + 0.5
    # distance from r_aim the entrance slit of the analyzer
    dist_s = dist_A4 + 0.5

    # coordinates of the center of the ALPHA3 plates
    geom.add_coords('A3', 'aim', dist_A3, geom.sec_angles)

    # coordinates of the center of the BETA3 plates
    geom.add_coords('B3', 'aim', dist_B3, geom.sec_angles)

    # coordinates of the center of the ALPHA4 plates
    geom.add_coords('A4', 'aim', dist_A4, geom.sec_angles)

    # Coordinates of the CENTRAL slit
    geom.add_coords('slit', 'aim', dist_s, geom.sec_angles)

    # Coordinates of the ANALYZER
    geom.add_coords('an', 'aim', dist_s, geom.sec_angles)

    # print info
    print('\nShot parameters: Btor = {} T, Ipl = {} MA'. format(Btor, Ipl))
    print('Primary beamline angles: ', geom.prim_angles[0:2])
    print('Secondary beamline angles: ', geom.sec_angles[0:3])
    print('r0 = ', np.round(geom.r_dict['r0'], 3))
    print('r_aim = ', r_aim)
    print('r_slit = ', np.round(geom.r_dict['slit'], 3))

    # TOKAMAK GEOMETRY
    # chamber entrance and exit coordinates
    geom.chamb_ent = [(2.016, 1.069), (2.238, 1.193),
                      (2.211, 0.937), (2.363, 1.04)]
    geom.chamb_ext = [(2.39, -0.44), (2.39, -2.0),
                      (2.39, 0.44), (2.39, 0.8)]

    # Toroidal Field coil
    geom.coil = np.loadtxt('TFCoil.dat') / 1000  # [m]
    # Poloidal Field coils
    geom.pf_coils = hb.import_PFcoils('PFCoils.dat')
    # Camera contour
    geom.camera = np.loadtxt('T15_vessel.txt') / 1000
    # Separatrix contour
    geom.sep = np.loadtxt('T15_sep.txt') / 1000
    # First wall innner and outer contours
    geom.in_fw = np.loadtxt('infw.txt') / 1000  # [m]
    geom.out_fw = np.loadtxt('outfw.txt') / 1000  # [m]

    return geom
