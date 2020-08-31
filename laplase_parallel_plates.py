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

    # beamline = 'prim'
    beamline = 'sec'
    save_data = True
    # define plates geometry
    # initially plates are parallel to XZ plane
    length = 0.4  # along X [m]
    width = 0.2  # along Z [m]
    thick = 0.02  # [m]
    gap = 0.2  # distance between plates along Y [m]
    plts_geom = np.array([length, width, thick, gap])

    # define center position
    plts_center = np.array([0., 0., 0.])  # plates center
    alpha = 45.  # angle with X axis in XY plane (alpha)
    beta = 20.  # angle with X axis in XZ plane (beta)
    gamma0 = -20.
    # gamma 0 for A2, -90 for B2
    gamma = gamma0 - 90.  # -90. # angle of rotation around X axis (gamma)
    # convert degrees to radians
    drad = np.pi/180.
    plts_angles = np.array([alpha, beta, gamma])

    # Create mesh grid
    # length of the X-edge of the domain [m]
    border_x = round(2*length*np.cos(alpha*drad)*np.cos(beta*drad), 2)
    border_z = round(2*(width + abs(length*np.sin(beta*drad))), 2)
    border_y = round(2*(gap + abs(length*np.sin(alpha*drad))), 2)
    delta = thick/2  # space step
    domain = np.array([border_x, border_y, border_z, delta])

    range_x = np.arange(-border_x/2., border_x/2., delta)
    range_y = np.arange(-border_y/2., border_y/2., delta)
    range_z = np.arange(-border_z/2., border_z/2., delta)
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
    Uupper_plate = 0.
    Ulower_plate = 1e3

    # array for electric potential
    U = np.zeros((mx, my, mz))

    U0 = np.copy(U)
    U1 = np.full_like(U, 1e3)

    UP_rotated, LP_rotated, upper_plate_flag, lower_plate_flag = \
        hb.plate_flags(range_x, range_y, range_z, U, plts_geom, plts_angles)

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
        if abs(abs(gamma) - abs(gamma0)) > 1e-2 and beamline == 'prim':
            plts_name = 'B2'
        if abs(abs(gamma) - abs(gamma0)) < 1e-2 and beamline == 'prim':
            plts_name = 'A2'
        if abs(abs(gamma) - abs(gamma0)) > 1e-2 and beamline == 'sec':
            plts_name = 'B3'
        if abs(abs(gamma) - abs(gamma0)) < 1e-2 and beamline == 'sec':
            plts_name = 'A3'
        hb.save_E(beamline, plts_name, Ex, Ey, Ez,
                  plts_angles, plts_geom, domain,
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
