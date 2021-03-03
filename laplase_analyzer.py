import numpy as np
import time
import hibplib as hb
import hibpplotlib as hbplot

'''
Calculate electric potential and electric field between two plates
'''


# %%
def pde_solve_full(U, Uupper_plate, Ulower_plate, upper_plate_flag,
                   lower_plate_flag, edge_flag, eps=1e-5):
    '''solve Laplase equation
    '''
    U0 = np.copy(U)
    U1 = np.full_like(U, 1e3)
    step = 0

    while np.amax(np.abs(U1-U0)) > eps:
        step += 1
        U0 = np.copy(U)

        # apply initial conditions at every time step
        # plates conditions
        U[upper_plate_flag] = Uupper_plate
        U[lower_plate_flag] = Ulower_plate
        # boundary conditions
        U[edge_flag] = 0.0

        U[1:-1, 1:-1, 1:-1] = (U[0:-2, 1:-1, 1:-1] + U[2:, 1:-1, 1:-1] +
                               U[1:-1, 0:-2, 1:-1] + U[1:-1, 2:, 1:-1] +
                               U[1:-1, 1:-1, 0:-2] + U[1:-1, 1:-1, 2:])/6.

        if step > 1000:  # wait until potential spreads to the center point
            U1 = np.copy(U)
#            print(np.amax(np.abs(U1-U0)))

    print('Total number of steps = {}'.format(step))
    return U


# %%
if __name__ == '__main__':

    plts_name = 'A4'
    save_data = True

    # define voltages [Volts]
    Uupper_plate = 0.
    Ulower_plate = 1e3

    # define center position
    plts_center = np.array([0., 0., 0.])  # plates center
    # initially plates are parallel to XZ plane
    # define primary beamline angles
    alpha_prim = 20.  # angle with X axis in XY plane (alpha)
    beta_prim = -10.  # angle with X axis in XZ plane (beta)
    gamma_prim = 0.  # rotation around the X axis (gamma)
    # define secondary beamline angles
    alpha_sec = 20.  # angle with X axis in XY plane (alpha)
    beta_sec = 20.  # angle with X axis in XZ plane (beta)
    gamma_sec = -20.  # rotation around the X axis (gamma)

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
        alpha, beta, gamma = alpha_prim, beta_prim, gamma_prim

    elif plts_name == 'B2':
        beamline = 'prim'
        length = 0.2  # along X [m]
        width = 0.08  # along Z [m]
        thick = 0.008  # [m]
        gap = 0.05  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_prim, beta_prim, gamma_prim-90.

    elif plts_name == 'A3':
        beamline = 'sec'
        length = 0.6  # along X [m]
        width = 0.2  # along Z [m]
        thick = 0.02  # [m]
        gap = 0.2  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_sec, beta_sec, gamma_sec

    elif plts_name == 'B3':
        beamline = 'sec'
        length = 0.4  # along X [m]
        width = 0.2  # along Z [m]
        thick = 0.02  # [m]
        gap = 0.2  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_sec, beta_sec, gamma_sec-90.

    elif plts_name == 'A4':
        beamline = 'sec'
        length = 0.4  # along X [m]
        width = 0.2  # along Z [m]
        thick = 0.02  # [m]
        gap = 0.2  # distance between plates along Y [m]
        alpha, beta, gamma = alpha_sec, beta_sec, gamma_sec

    elif plts_name == 'an':
        # ANALYZER
        # define voltages [Volts]
        Uupper_plate = 1e3
        Ulower_plate = 0.
        beamline = 'sec'
        # slits configuration [m]
        n_slits, slit_dist, slit_w = 7, 0.01, 5e-3
        # analyzer geometry
        theta_an = 30.
        width = 0.2  # along Z [m]
        thick = 0.02  # [m]
        gap = 0.1  # distance between plates along Y [m]

        YD1 = 0.02 + thick + np.cos(theta_an*drad) * (n_slits//2 * slit_dist
                                                      + 0.5*slit_w)
        YD2 = YD1
        YD = YD1 + YD2
        XD = 3 * np.sqrt(3) * YD

        length = 1.2*XD  # along X [m]

        alpha, beta, gamma = alpha_sec-theta_an, beta_sec, gamma_sec

        # G coeff of the analyzer
        G = (XD*np.tan(theta_an*drad) - YD) / (4 * gap *
                                               np.sin(theta_an*drad)**2)
        G = np.round(G, 5)

        # center of the coords system should be shifted to the slit center
        plts_center = np.array([XD/2, gap/2 + YD1, 0])
        axis = hb.calc_vector(1, alpha, beta)
        plts_center = hb.rotate(plts_center, axis=(0, 0, 1), deg=alpha)
        plts_center = hb.rotate(plts_center, axis=(0, 1, 0), deg=beta)
        plts_center = hb.rotate(plts_center, axis=axis, deg=gamma)

        an_params = np.array([n_slits, slit_dist, slit_w, G, theta_an,
                              round(XD, 4), round(YD1, 4), round(YD2, 4)])
        print('\n ANALYZER with {} slits is defined'.format(n_slits))
        print('\n G = {}\n'.format(G))

    # Create mesh grid
    # length of the X-edge of the domain [m]
    border_x = round(2*length*np.cos(alpha*drad)*np.cos(beta*drad), 2)
    border_z = round(2*(width + abs(length*np.sin(beta*drad))), 2)
    border_y = round(2*(gap + 2*thick + abs(length*np.sin(alpha*drad))), 2)
    delta = thick/2  # space step

    range_x = np.arange(-border_x/2., border_x/2., delta) + plts_center[0]
    range_y = np.arange(-border_y/2., border_y/2., delta) + plts_center[1]
    range_z = np.arange(-border_z/2., border_z/2., delta) + plts_center[2]
    x, y, z = np.meshgrid(range_x, range_y,
                          range_z, indexing='ij')  # [X ,Y, Z]
    # collect xmin, xmax, ymin, ymax, zmin, zmax, delta
    domain = np.array([range_x[0], range_x[-1]+delta,
                       range_y[0], range_y[-1]+delta,
                       range_z[0], range_z[-1]+delta, delta])

    mx = range_x.shape[0]
    my = range_y.shape[0]
    mz = range_z.shape[0]

    # define mask for edge elements
    edge_flag = np.full_like(x, False, dtype=bool)
    edge_list = [0, 1]  # numbers of edge elements
    edge_flag[edge_list, :, :] = True
    edge_flag[:, edge_list, :] = True
    edge_flag[:, :, edge_list] = True

    # array for electric potential
    U = np.zeros((mx, my, mz))

    plts_angles = np.array([alpha, beta, gamma])
    plts_geom = np.array([length, width, thick, gap])

    # print info
    print('Solving for ' + plts_name)
    print('Geom: ', plts_geom)
    print('Angles: ', plts_angles)

    UP_rotated, LP_rotated, upper_plate_flag, lower_plate_flag = \
        hb.plate_flags(range_x, range_y, range_z, U,
                       plts_geom, plts_angles, plts_center)

# %% solver
    t1 = time.time()

    # calculation loop
    U = pde_solve_full(U, Uupper_plate, Ulower_plate, upper_plate_flag,
                       lower_plate_flag, edge_flag, eps=1e-5)

    t2 = time.time()
    print("time needed for calculation: {:.5f} s\n".format(t2-t1))

# %% save electric field
    Ex, Ey, Ez = np.gradient(-1*U, delta)  # Ex, Ey, Ez
    # set zero E in the cells corresponding to plates
    Ex[upper_plate_flag], Ey[upper_plate_flag], Ez[upper_plate_flag] = 0, 0, 0
    Ex[lower_plate_flag], Ey[lower_plate_flag], Ez[lower_plate_flag] = 0, 0, 0
    if save_data and beamline == 'prim':
        hb.save_E(beamline, plts_name, Ex, Ey, Ez,
                  np.array([alpha_prim, beta_prim, gamma_prim]),
                  plts_geom, domain, an_params,
                  UP_rotated[4:], LP_rotated[4:])
    elif save_data and beamline == 'sec':
        hb.save_E(beamline, plts_name, Ex, Ey, Ez,
                  np.array([alpha_sec, beta_sec, gamma_sec]),
                  plts_geom, domain, an_params,
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
