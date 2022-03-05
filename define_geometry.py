'''
Define main parameters of T-15MD HIBP geometry
'''
import numpy as np
import hibplib as hb


# %%
def define_geometry(analyzer=1):
    '''

    Parameters
    ----------
    analyzer : int, optional
        number of the HIBP Analyzer. The default is 1.

    Returns
    -------
    geom : Geometry
        Geometry object with a proper configuration.

    '''
    geom = hb.Geometry()

    # plasma parameters
    geom.R = 1.5  # tokamak major radius [m]
    geom.r_plasma = 0.7  # plasma minor radius [m]
    geom.elon = 1.8  # plasma elongation

    # PRIMARY beamline geometry
    # alpha and beta angles of the PRIMARY beamline [deg]
    alpha_prim = 34.  # 20.  # 30.
    beta_prim = -10.
    gamma_prim = 0.
    prim_angles = {'r0': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'B2': np.array([alpha_prim, beta_prim, gamma_prim]),
                   'A2': np.array([alpha_prim, beta_prim, gamma_prim])}
    geom.angles_dict.update(prim_angles)

    # coordinates of the injection port [m]
    xport_in = 2.136  # 1.5 + 0.726
    yport_in = 1.189  # 1.064
    zport_in = 0.019  # 0.0
    geom.r_dict['port_in'] = np.array([xport_in, yport_in, zport_in])

    # distance from the injection port to the Alpha2 plates
    dist_A2 = 0.3  # [m]
    # distance from Alpha2 plates to the Beta2 plates
    dist_B2 = 0.4  # [m]
    # distance from Beta2 plates to the initial point of the traj [m]
    dist_r0 = 0.2

    # coordinates of the center of the ALPHA2 plates
    geom.add_coords('A2', 'port_in', dist_A2, geom.angles_dict['A2'])
    # coordinates of the center of the BETA2 plates
    geom.add_coords('B2', 'A2', dist_B2, geom.angles_dict['B2'])
    # coordinates of the initial point of the trajectory [m]
    geom.add_coords('r0', 'B2', dist_r0, geom.angles_dict['r0'])

    # AIM position (BEFORE the Secondary beamline) [m]
    if analyzer == 1:
        xaim = 2.6  # 2.5
        yaim = -0.1  # 0.0  # -0.15  # -0.25
        zaim = zport_in  # 0.0
        # alpha and beta angles of the SECONDARY beamline [deg]
        alpha_sec = 10.
        beta_sec = 20.
        gamma_sec = 0.  # -20.
        A3_angles = np.array([alpha_sec, beta_sec, gamma_sec])
    elif analyzer == 2:
        xaim = 2.6  # 2.5
        yaim = 0.0  # 0.15
        zaim = zport_in  # 0.0
        # alpha and beta angles of the SECONDARY beamline [deg]
        alpha_sec = 30.  # 35.  # 5.
        beta_sec = 20.  # 25.
        gamma_sec = 0.  # -20.
        # in the second line U_lower_plate=0
        A3_angles = np.array([alpha_sec, beta_sec, gamma_sec+180.])
    r_aim = np.array([xaim, yaim, zaim])
    geom.r_dict['aim'] = r_aim

    # SECONDARY beamline geometry
    sec_angles = {'A3': A3_angles,
                  'B3': np.array([alpha_sec, beta_sec, gamma_sec]),
                  'A4': np.array([alpha_sec, beta_sec, gamma_sec]),
                  'an': np.array([alpha_sec, beta_sec, gamma_sec])}
    geom.angles_dict.update(sec_angles)

    # distance from r_aim to the Alpha3 center
    dist_A3 = 0.2  # 0.3  # 1/2 of plates length
    # distance from Alpha3 to the Beta3 center
    dist_B3 = 0.5  # + 0.6
    # from Beta3 to Alpha4
    dist_A4 = 0.5
    # distance from Alpha4 to the entrance slit of the analyzer
    dist_s = 0.5

    # coordinates of the center of the ALPHA3 plates
    geom.add_coords('A3', 'aim', dist_A3, geom.angles_dict['A3'])
    # coordinates of the center of the BETA3 plates
    geom.add_coords('B3', 'A3', dist_B3, geom.angles_dict['B3'])
    # coordinates of the center of the ALPHA4 plates
    geom.add_coords('A4', 'B3', dist_A4, geom.angles_dict['A4'])
    # Coordinates of the CENTRAL slit
    geom.add_coords('slit', 'A4', dist_s, geom.angles_dict['an'])
    # Coordinates of the ANALYZER
    geom.add_coords('an', 'A4', dist_s, geom.angles_dict['an'])

    # print info
    print('\nDefining geometry for Analyzer #{}'.format(analyzer))
    print('\nPrimary beamline angles: ', geom.angles_dict['r0'])
    print('Secondary beamline angles: ', geom.angles_dict['A3'])
    print('r0 = ', np.round(geom.r_dict['r0'], 3))
    print('r_aim = ', np.round(geom.r_dict['aim'], 3))
    print('r_slit = ', np.round(geom.r_dict['slit'], 3))

    # TOKAMAK GEOMETRY
    # chamber entrance and exit coordinates
    geom.chamb_ent = [(1.87, 1.152), (1.675, 1.434),
                      (2.358, 0.445), (1.995, 0.970)]
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
