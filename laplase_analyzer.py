import numpy as np
import time
import hibplib as hb
import hibpplotlib as hbplot

'''
Calculate electric potential and electric field between two plates
'''


# %%
def init_conditions(U, Uupper_plate, Ulower_plate, upper_plate_flag,
                    lower_plate_flag, edge_flag):
    # plates conditions
    U[upper_plate_flag] = Uupper_plate
    U[lower_plate_flag] = Ulower_plate
    # boundary conditions
    U[edge_flag] = 0.0
    return U


# %%
def pde_step(U, Uupper_plate, Ulower_plate, upper_plate_flag,
             lower_plate_flag, edge_flag):
    '''
    Partial Differential Equation calculation at a single time step t
    '''
    # apply initial conditions at every time step
    U = init_conditions(U, Uupper_plate, Ulower_plate, upper_plate_flag,
                        lower_plate_flag, edge_flag)

    U[1:-1, 1:-1, 1:-1] = (U[0:-2, 1:-1, 1:-1] + U[2:, 1:-1, 1:-1] +
                           U[1:-1, 0:-2, 1:-1] + U[1:-1, 2:, 1:-1] +
                           U[1:-1, 1:-1, 0:-2] + U[1:-1, 1:-1, 2:])/6.
    return U


# %%
if __name__ == '__main__':

    plts_name = 'an'
    save_data = True

    # define center position
    plts_center = np.array([0., 0., 0.])  # plates center
    # initially plates are parallel to XZ plane
    # define primary beamline angles
    alpha_prim = 25.  # angle with X axis in XY plane (alpha)
    beta_prim = -10.  # angle with X axis in XZ plane (beta)
    gamma0_prim = 0.  # rotation around the X axis (gamma)
    # define secondary beamline angles
    alpha_sec = 20.  # angle with X axis in XY plane (alpha)
    beta_sec = 20.  # angle with X axis in XZ plane (beta)
    gamma0_sec = -20.  # rotation around the X axis (gamma)

    # convert degrees to radians
    drad = np.pi/180.
    # analyzer parameters
    an_params = None

    # define plates geometry
    if plts_name == 'A2':
        beamline = 'prim'
        length = 0.2  # along X [m]
        width = 0.08  # along Z [m]
        thick = 0.008  # [m]
        gap = 0.05  # distance between plates along Y [m]
        # gamma 0 for A2, -90 for B2
        alpha, beta, gamma = alpha_prim, beta_prim, gamma0_prim
    if plts_name == 'B2':
        beamline = 'prim'
        length = 0.2  # along X [m]
        width = 0.08  # along Z [m]
        thick = 0.008  # [m]
        gap = 0.05  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_prim, beta_prim, gamma0_prim-90.
    if plts_name == 'A3':
        beamline = 'sec'
        length = 0.6  # along X [m]
        width = 0.2  # along Z [m]
        thick = 0.02  # [m]
        gap = 0.2  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_sec, beta_sec, gamma0_sec
    if plts_name == 'B3':
        beamline = 'sec'
        length = 0.4  # along X [m]
        width = 0.2  # along Z [m]
        thick = 0.02  # [m]
        gap = 0.2  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_sec, beta_sec, gamma0_sec-90.
    if plts_name == 'an':
        # ANALYZER
        beamline = 'sec'
        # slits configuration [m]
        n_slits, slit_dist, slit_w = 7, 0.01, 5e-3
        # analyzer geometry
        theta_an = 30.
        YD1 = 1.5 * np.cos(theta_an*drad) * (n_slits//2 * slit_dist
                                             + 0.5*slit_w)
        # YD1 = 1.2 * np.cos(theta_an*drad) * (n_slits//2 * (slit_w + slit_dist)
        #                                      + 0.5*slit_w)
        YD2 = YD1
        YD = YD1 + YD2
        XD = 3 * np.sqrt(3) * YD

        length = 1.2*XD  # along X [m]
        width = 0.2  # along Z [m]
        thick = 0.02  # [m]
        gap = 0.15  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_sec-theta_an, beta_sec, gamma0_sec

        # G coeff of the analyzer
        G = (XD*np.tan(theta_an*drad) - YD) / (4 * gap *
                                               np.sin(theta_an*drad)**2)
        G = np.round(G, 5)

        # center of the coords system should be shifted to the slit center
        # distance from coords center to slit center
        dist = np.sqrt((XD/2)**2 + (gap/2 + YD1)**2)
        # alpha angle of the vector
        alpha_shift = np.arctan((YD1 + gap/2) / (XD/2)) / drad  # [deg]
        # resulting shift vector
        zero_shift = hb.calc_vector(dist, alpha_shift, beta)
        plts_center += zero_shift

        an_params = np.array([n_slits, slit_dist, slit_w, G, theta_an,
                              round(XD, 4), round(YD1, 4), round(YD2, 4)])
        print('\n ANALYZER with {} slits is defined'.format(n_slits))
        print('\n G = {}\n'.format(G))

    plts_angles = np.array([alpha, beta, gamma])
    plts_geom = np.array([length, width, thick, gap])

    # print info
    print('Solving for ' + plts_name)
    print('Geom: ', plts_geom)
    print('Angles: ', plts_angles)

    # Create mesh grid
    # length of the X-edge of the domain [m]
    border_x = round(2*length*np.cos(alpha*drad)*np.cos(beta*drad), 2)
    border_z = round(2*(width + abs(length*np.sin(beta*drad))), 2)
    border_y = round(2*(gap + abs(length*np.sin(alpha*drad))), 2)
    delta = thick/2  # space step
    domain = np.array([border_x, border_y, border_z, delta])

    range_x = np.arange(-border_x/2., border_x/2., delta) + plts_center[0]
    range_y = np.arange(-border_y/2., border_y/2., delta) + plts_center[1]
    range_z = np.arange(-border_z/2., border_z/2., delta) + plts_center[2]
    x, y, z = np.meshgrid(range_x, range_y,
                          range_z, indexing='ij')  # [X ,Y, Z]

    mx = range_x.shape[0]
    my = range_y.shape[0]
    mz = range_z.shape[0]

    # define mask for edge elements
    edge_flag = np.full_like(x, False, dtype=bool)
    edge_list = [0, 1]  # numbers of edge elements
    edge_flag[edge_list, :, :] = True
    edge_flag[:, edge_list, :] = True
    edge_flag[:, :, edge_list] = True

    # define voltages [Volts]
    Uupper_plate = 1e3
    Ulower_plate = 0.

    # array for electric potential
    U = np.zeros((mx, my, mz))

    U0 = np.copy(U)
    U1 = np.full_like(U, 1e3)

    UP_rotated, LP_rotated, upper_plate_flag, lower_plate_flag = \
        hb.plate_flags(range_x, range_y, range_z, U,
                       plts_geom, plts_angles, plts_center)

# %% solver
    eps = 1e-5

    t1 = time.time()
    # calculation loop
    step = 0

    while np.amax(np.abs(U1-U0)) > eps:
        step += 1
        U0 = np.copy(U)
        U = pde_step(U, Uupper_plate, Ulower_plate, upper_plate_flag,
                     lower_plate_flag, edge_flag)
        if step > 1000:  # wait until potential spreads to the center point
            U1 = np.copy(U)
#            print(np.amax(np.abs(U1-U0)))

    print('Total number of steps = {}'.format(step))
    t2 = time.time()
    print("time needed for calculation: {:.5f} s\n".format(t2-t1))

# %% save electric field
    Ex, Ey, Ez = np.gradient(-1*U, delta)  # Ex, Ey, Ez
    if save_data:
        if plts_name == 'an':
            plts_angles[0] += theta_an
        hb.save_E(beamline, plts_name, Ex, Ey, Ez,
                  plts_angles, plts_geom, domain, an_params,
                  UP_rotated[4:], LP_rotated[4:])
    else:
        print('DATA NOT SAVED')

# %% plot results
    hbplot.plot_contours(range_x, range_y, range_z, U,
                         upper_plate_flag, lower_plate_flag, 30)
    hbplot.plot_stream(range_x, range_y, range_z, Ex, Ey, Ez,
                       upper_plate_flag, lower_plate_flag, dens=1.0)
#    plot_quiver(range_x, range_y, range_z, Ex, Ey, Ez)
#    plot_quiver3d(x, y, z, Ex, Ey, Ez, 6)
