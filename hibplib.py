'''
Heavy Ion Beam Probe partile tracing library
'''
import numpy as np
import os
import errno
import pickle as pc
import hibpplotlib as hbplot
import copy
from matplotlib import path
from scipy.interpolate import RegularGridInterpolator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from itertools import cycle
import numba


# %% define class for trajectories

class Traj():
    '''
    Trajectory object
    '''

    def __init__(self, q, m, Ebeam, r0, alpha, beta, U, dt=1e-7):
        '''

        Parameters
        ----------
        q : float
            particle charge [Co]
        m : float
            particle mass [kg]
        Ebeam : float
            beam energy [keV]
        r0 : np.array
            initial point of the trajectory [m]
        alpha : float
            injection angle in XY plane [rad]
        beta : float
            injection angle in XZ plane [rad]
        U : dict
            dict of voltages in [kV] keys=[A1 B1 A2 B2 A3 B3 an]
        dt : float, optional
            timestep for RK algorithm [s]. The default is 1e-7.

        Returns
        -------
        None.

        '''
        self.q = q
        self.m = m
        self.Ebeam = Ebeam
        # particle velocity:
        Vabs = np.sqrt(2 * Ebeam * 1.602176634E-16 / m)
        V0 = calc_vector(Vabs, alpha, beta, direction=(-1, -1, 1))
        self.alpha = alpha
        self.beta = beta
        self.U = U
        self.RV0 = np.array([np.hstack((r0, V0))])  # initial condition
        # array with r,V for the primary trajectory
        self.RV_prim = self.RV0
        self.tag_prim = [1]
        # array with r,V for the secondary trajectory
        self.RV_sec = np.array([[]])
        self.tag_sec = [2]
        # list to contain RV of the whole fan:
        self.Fan = []
        # time step for primary orbit:
        self.dt1 = dt
        # time step for secondary orbit:
        self.dt2 = dt
        # flags
        self.IsAimXY = False
        self.IsAimZ = False
        self.fan_ok = False
        self.IntersectGeometry = {'A2': False, 'B2': False, 'chamb': False}
        self.IntersectGeometrySec = {'A3': False, 'B3': False, 'A4': False,
                                     'chamb': False}
        self.B_out_of_bounds = False
        # multislit:
        self.fan_to_slits = []
        self.RV_sec_toslits = []
        self.ion_zones = []

    def pass_prim(self, E_interp, B_interp, geom, tmax=1e-5):
        '''
        passing primary trajectory from initial point self.RV0
        E_interp : dictionary with E field interpolants
        B_interp : list with B fied interpolants
        geom : Geometry object
        '''
        print('\n Passing primary trajectory')
        # reset intersection flags
        for key in self.IntersectGeometry.keys():
            self.IntersectGeometry[key] = False
        t = 0.
        dt = self.dt1
        RV_old = self.RV0  # initial position
        RV = self.RV0  # array to collect all r, V
        k = self.q / self.m
        tag_column = [10]

        while t <= tmax:
            r = RV_old[0, :3]
            # Electric field:
            E_local = return_E(r, E_interp, self.U, geom)
            # Magnetic field:
            try:
                B_local = return_B(r, B_interp)
            except ValueError:
                print('Btor Out of bounds for primaries, r = ', r)
                print(' t = ', t)
                break
            # runge-kutta step:
            RV_new = runge_kutt(k, RV_old, dt, E_local, B_local)
            RV = np.vstack((RV, RV_new))

            tag_column = np.hstack((tag_column, 10))

            if geom.check_chamb_intersect('prim', RV_old[0, 0:3],
                                          RV_new[0, 0:3]):
                print('Primary intersected chamber')
                self.IntersectGeometry['chamb'] = True
                break

            plts_flag, plts_name = geom.check_plates_intersect(RV_old[0, 0:3],
                                                               RV_new[0, 0:3])
            if plts_flag:
                print('Primary intersected ' + plts_name + ' plates')
                self.IntersectGeometry[plts_name] = True
                break

            RV_old = RV_new
            t = t + dt
            # print('t = ', t)

        self.RV_prim = RV
        self.tag_prim = tag_column

    def pass_sec(self, RV0, r_aim, E_interp, B_interp, geom,
                 stop_plane_n=np.array([1, 0, 0]), tmax=5e-5,
                 eps_xy=1e-3, eps_z=1e-3):
        '''
        passing secondary trajectory from initial point RV0 to point r_aim
        with accuracy eps
        RV0 : initial position and velocity
        '''
        # print('Passing secondary trajectory')
        self.IsAimXY = False
        self.IsAimZ = False
        self.B_out_of_bounds = False
        # reset intersection flags
        for key in self.IntersectGeometrySec.keys():
            self.IntersectGeometrySec[key] = False
        t = 0.
        dt = self.dt2
        RV_old = RV0  # initial position
        RV = RV0  # array to collect all [r,V]
        k = 2*self.q / self.m
        tag_column = [20]

        while t <= tmax:  # to witness curls initiate tmax as 1 sec
            r = RV_old[0, :3]
            # Electric field:
            E_local = return_E(r, E_interp, self.U, geom)
            # Magnetic field:
            try:
                B_local = return_B(r, B_interp)
            except ValueError:
                print('Btor Out of bounds for secondaries, r = ',
                      np.round(r, 3))
                print(' t = ', t)
                self.B_out_of_bounds = True
                break
            # runge-kutta step:
            RV_new = runge_kutt(k, RV_old, dt, E_local, B_local)

            if geom.check_chamb_intersect('sec', RV_old[0, 0:3],
                                          RV_new[0, 0:3]):
                # print('Secondary intersected chamber exit')
                self.IntersectGeometrySec['chamb'] = True

            plts_flag, plts_name = geom.check_plates_intersect(RV_old[0, 0:3],
                                                               RV_new[0, 0:3])
            if plts_flag:
                print('Secondary intersected ' + plts_name + ' plates')
                self.IntersectGeometrySec[plts_name] = True

            # find last point of the secondary trajectory
            if (RV_new[0, 0] > 2.5) and (RV_new[0, 1] < 1.5):
                # intersection with the stop plane:
                r_intersect = line_plane_intersect(stop_plane_n, r_aim,
                                                   RV_new[0, :3]-RV_old[0, :3],
                                                   RV_new[0, :3])
                # check if r_intersect is between RV_old and RV_new:
                if is_between(RV_old[0, :3], RV_new[0, :3], r_intersect):
                    RV_new[0, :3] = r_intersect
                    RV = np.vstack((RV, RV_new))
                    # check XY plane:
                    if (np.linalg.norm(RV_new[0, :2] - r_aim[:2]) <= eps_xy):
                        # print('aim XY!')
                        self.IsAimXY = True
                    # check XZ plane:
                    if (np.linalg.norm(RV_new[0, [0, 2]] - r_aim[[0, 2]]) <=
                            eps_z):
                        # print('aim Z!')
                        self.IsAimZ = True
                    break

            # continue trajectory calculation:
            RV_old = RV_new
            t = t + dt
            RV = np.vstack((RV, RV_new))
            tag_column = np.hstack((tag_column, 20))
            # print('t secondary = ', t)

        self.RV_sec = RV
        self.tag_sec = tag_column

    def pass_fan(self, r_aim, E_interp, B_interp, geom,
                 stop_plane_n=np.array([1, 0, 0]), eps_xy=1e-3, eps_z=1e-3,
                 no_intersect=False, no_out_of_bounds=False):
        '''
        passing fan from initial point self.RV0
        '''
        print('\n Passing fan of trajectories')
        self.pass_prim(E_interp, B_interp, geom)
        # create a list fro secondary trajectories:
        list_sec = []
        # check intersection of primary trajectory:
        if True in self.IntersectGeometry.values():
            print('Fan list is empty')
            self.Fan = list_sec
            return

        # check eliptical radius of particle:
        # 1.5 m - major radius of a torus, elon - size along Y
        mask = np.sqrt((self.RV_prim[:, 0] - geom.R)**2 +
                       (self.RV_prim[:, 1] / geom.elon)**2) <= geom.r_plasma
        self.tag_prim[mask] = 11

        # list of initial points of secondary trajectories:
        RV0_sec = self.RV_prim[(self.tag_prim == 11)]

        for RV02 in RV0_sec:
            RV02 = np.array([RV02])
            self.pass_sec(RV02, r_aim, E_interp, B_interp, geom,
                          stop_plane_n=stop_plane_n,
                          eps_xy=eps_xy, eps_z=eps_z)
            if (no_intersect and True in self.IntersectGeometrySec.values()) or \
               (no_out_of_bounds and self.B_out_of_bounds):
                continue
            list_sec.append(self.RV_sec)

        self.Fan = list_sec

    def pass_to_target(self, r_aim, E_interp, B_interp, geom,
                       stop_plane_n=np.array([1, 0, 0]),
                       eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10,
                       no_intersect=False, no_out_of_bounds=False):
        '''
        find secondary trajectory which goes directly to target
        '''
        if True in self.IntersectGeometry.values():
            print('There is intersection at primary trajectory')
            return
        if len(self.Fan) == 0:
            print('NO secondary trajectories')
            return
        # reset flags in order to let the algorithm work properly
        self.IsAimXY = False
        self.IsAimZ = False
        # reset intersection flags for secondaries
        for key in self.IntersectGeometrySec.keys():
            self.IntersectGeometrySec[key] = False

        # find which secondaries are higher/lower than r_aim
        # sign = -1 means higher, 1 means lower
        signs = np.array([np.sign(np.cross(RV[-1, :3], r_aim)[-1])
                          for RV in self.Fan])
        are_higher = np.argwhere(signs == -1)
        are_lower = np.argwhere(signs == 1)
        twisted_fan = False  # flag to detect twist of the fan

        if are_higher.shape[0] == 0:
            print('all secondaries are lower than aim!')
            n = int(are_lower[are_lower.shape[0]//2])
        elif are_lower.shape[0] == 0:
            print('all secondaries are higher than aim!')
            n = int(are_higher[are_higher.shape[0]//2])
        else:
            if are_higher[-1] > are_lower[0]:
                print('Fan is twisted!')
                twisted_fan = True
                n = int(are_lower[-1])
            else:
                n = int(are_higher[-1])  # find the last one which is higher
                self.fan_ok = True
        RV_old = np.array([self.Fan[n][0]])

        # find secondary, which goes directly into r_aim
        self.dt1 = self.dt1/2.
        while True:
            # make a small step along primary trajectory
            r = RV_old[0, :3]
            try:
                B_local = return_B(r, B_interp)
            except ValueError:
                print('B out of bounds while passing secondary to target')
                break
            E_local = np.array([0., 0., 0.])
            RV_new = runge_kutt(self.q / self.m, RV_old, self.dt1,
                                E_local, B_local)
            # pass new secondary trajectory
            self.pass_sec(RV_new, r_aim, E_interp, B_interp, geom,
                          stop_plane_n=stop_plane_n,
                          eps_xy=eps_xy, eps_z=eps_z)
            # check XY flag
            if self.IsAimXY:
                # insert RV_new into primary traj
                # find the index of the point in primary traj closest to RV_new
                ind = np.nanargmin(np.linalg.norm(self.RV_prim[:, :3] -
                                                  RV_new[0, :3], axis=1))
                if is_between(self.RV_prim[ind, :3],
                              self.RV_prim[ind+1, :3], RV_new[0, :3], eps=1e-4):
                    i2insert = ind+1
                else:
                    i2insert = ind
                self.RV_prim = np.insert(self.RV_prim, i2insert, RV_new, axis=0)
                self.tag_prim = np.insert(self.tag_prim, i2insert, 11, axis=0)
                break
            # check if the new secondary traj is lower than r_aim
            if (not twisted_fan and
                    np.sign(np.cross(self.RV_sec[-1, :3], r_aim)[-1]) > 0):
                # if lower, halve the timestep and try once more
                self.dt1 = self.dt1/2.
                print('dt1={}'.format(self.dt1))
                if self.dt1 < dt_min:
                    print('dt too small')
                    break
            else:
                # if higher, continue steps along the primary
                RV_old = RV_new

    def add_slits(self, n_slits):
        '''
        create empty list for secondary trajectories,
        which go to different slits
        '''
        if len(self.RV_sec_toslits) == n_slits:
            pass
        else:
            self.RV_sec_toslits = [None]*n_slits
            self.ion_zones = [None]*n_slits

    def plot_prim(self, ax, axes='XY', color='k', full_primary=False):
        '''
        plot primary trajectory
        '''
        axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
        index_X, index_Y = axes_dict[axes]
        index = -1
        if not full_primary:
            # find where secondary trajectory starts:
            for i in range(self.RV_prim.shape[0]):
                if np.linalg.norm(self.RV_prim[i, :3]
                                  - self.RV_sec[0, :3]) < 1e-4:
                    index = i+1
        ax.plot(self.RV_prim[:index, index_X],
                self.RV_prim[:index, index_Y],
                color=color, linewidth=2)

    def plot_sec(self, ax, axes='XY', color='r'):
        '''
        plot secondary trajectory
        '''
        axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
        index_X, index_Y = axes_dict[axes]
        ax.plot(self.RV_sec[:, index_X], self.RV_sec[:, index_Y],
                color=color, linewidth=2)

    def plot_fan(self, ax, axes='XY', color='r'):
        '''
        plot fan of secondary trajectories
        '''
        axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
        index_X, index_Y = axes_dict[axes]
        for i in self.Fan:
            ax.plot(i[:, index_X], i[:, index_Y], color=color)


# %% define class for plates
class Plates():
    '''
    object containing info on deflecting plates
    '''

    def __init__(self, name, beamline, r=np.array([0., 0., 0.])):
        '''

        Parameters
        ----------
        name : str
            plates name, 'A1', 'B1' etc
        beamline : str
            beamline name, 'prim' or 'sec'
        r : np.array, optional
            initial plates position. The default is np.array([0., 0., 0.]).

        Returns
        -------
        None.

        '''
        self.name = name
        self.beamline = beamline
        self.r = r

    def set_edges(self, edges):
        '''
        set coordinates of plates edges
        '''
        self.edges = edges

    def rotate(self, angles, beamline_angles, inverse=False):
        '''
        rotate plates on angles around the axis with beamline_angles
        '''
        self.angles = angles
        self.beamline_angles = beamline_angles
        for i in range(self.edges.shape[0]):
            self.edges[i, :] = rotate3(self.edges[i, :],
                                       angles, beamline_angles,
                                       inverse=inverse)

    def shift(self, r_new):
        '''
        shift all the coordinates to r_new
        '''
        self.r += r_new
        self.edges += r_new

    def check_intersect(self, point1, point2):
        '''
        check intersection with a segment point1 -> point2
        '''
        segment_coords = np.array([point1, point2])
        if segm_poly_intersect(self.edges[0][:4], segment_coords) or \
           segm_poly_intersect(self.edges[1][:4], segment_coords):
            return True
        # if plates are flared
        if self.edges.shape[1] > 4:
            # for flared plates the sequence of vertices is
            # [UP1sw, UP1, UP2, UP2sw, UP3, UP4]
            point_ind = [1, 4, 5, 2]
            if segm_poly_intersect(self.edges[0][point_ind], segment_coords) or \
               segm_poly_intersect(self.edges[1][point_ind], segment_coords):
                return True
        return False

    def plot(self, ax, axes='XY'):
        '''
        plot plates
        '''
        index_X, index_Y = get_index(axes)
        ax.fill(self.edges[0][:, index_X], self.edges[0][:, index_Y],
                fill=False, hatch='\\', linewidth=2)
        ax.fill(self.edges[1][:, index_X], self.edges[1][:, index_Y],
                fill=False, hatch='/', linewidth=2)


# %% class for Analyzer
class Analyzer(Plates):
    '''
    Analyzer object
    '''

    def add_slits(self, an_params):
        '''
        add slits and detector to Analyzer
        an_params : list containing [n_slits, slit_dist, slit_w, G, theta,
                                     XD, YD1, YD2]
        n_slits : number of slits
        slit_dist : distance between centers of the slits [m]
        slit_w : slit width (along Y) [m]
        theta : entrance angle to analyzer [deg]
        G : gain function
        XD, YD1, YD2 : geometry parameters [m]
        '''
        # define main parameters of the Analyzer
        self.n_slits, self.slit_dist, self.slit_w, self.G, self.theta, \
            self.XD, self.YD1, self.YD2 = an_params
        self.n_slits = int(self.n_slits)
        # length of the slit
        slit_l = 0.1
        # angles of the slits plane normal
        slit_angles = np.array([self.theta, 0., 0.])
        # coords of center of the central slit
        rs = np.array([0, 0, 0])
        # define slits
        r_slits, slit_plane_n, slits_spot = \
            define_slits(rs, slit_angles, self.n_slits, self.slit_dist,
                         self.slit_w, slit_l)
        # save slits edges
        self.slits_edges = r_slits
        self.slit_plane_n = slit_plane_n
        self.slits_spot = slits_spot
        # define detector
        n_det = self.n_slits
        # set detector angles
        det_angles = np.array([180. - self.theta, 0, 0])
        r_det, det_plane_n, det_spot = \
            define_slits(np.array([self.XD, self.YD1 - self.YD2, 0]),
                         det_angles, n_det, self.slit_dist, self.slit_dist,
                         slit_l)
        # save detector edges
        self.det_edges = r_det
        self.det_plane_n = det_plane_n
        self.det_spot = det_spot
        print('\nAnalyzer with {} slits ok!'.format(self.n_slits))
        print('G = {}'.format(self.G))

    def get_params(self):
        '''
        return analyzer parameters
        [n_slits, slit_dist, slit_w, G, theta, XD, YD1, YD2]
        '''
        print('n_slits = {}\nslit_dist = {}\nslit_width = {}'
              .format(self.n_slits, self.slit_dist, self.slit_w))
        print('G = {}\ntheta = {}\nXD = {}\nYD1 = {}\nYD2 = {}'
              .format(self.G, self.theta, self.XD, self.YD1, self.YD2))
        return(np.array([self.n_slits, self.slit_dist, self.slit_w, self.G,
                         self.theta, self.XD, self.YD1, self.YD2]))

    def rotate(self, angles, beamline_angles):
        '''
        rotate all the coordinates around the axis with beamline_angles
        '''
        super().rotate(angles, beamline_angles)
        for attr in [self.slits_edges, self.slits_spot,
                     self.det_edges, self.det_spot]:
            if len(attr.shape) < 2:
                attr = rotate3(attr, angles, beamline_angles)
            else:
                for i in range(attr.shape[0]):
                    attr[i, :] = rotate3(attr[i, :], angles, beamline_angles)
        # recalculate normal to slit plane:
        self.slit_plane_n = calc_normal(self.slits_edges[0, 0, :],
                                        self.slits_edges[0, 1, :],
                                        self.slits_edges[0, 2, :])
        self.det_plane_n = calc_normal(self.det_edges[0, 0, :],
                                       self.det_edges[0, 1, :],
                                       self.det_edges[0, 2, :])

    def shift(self, r_new):
        '''
        shift all the coordinates to r_new
        '''
        super().shift(r_new)
        for attr in [self.slits_edges, self.slits_spot,
                     self.det_edges, self.det_spot]:
            attr += r_new

    def plot(self, ax, axes='XY', n_slit='all'):
        '''
        plot analyzer
        '''
        # plot plates
        super().plot(ax, axes=axes)
        # choose which slits to plot
        index_X, index_Y = get_index(axes)
        if n_slit == 'all':
            slits = range(self.slits_edges.shape[0])
        else:
            slits = [n_slit]
        # set color cycler
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        colors = colors[:len(slits)]
        colors = cycle(colors)
        # plot slits and detector
        for edges, spot in zip([self.slits_edges, self.det_edges],
                               [self.slits_spot, self.det_spot]):
            for i in slits:
                c = next(colors)
                # plot center
                ax.plot(edges[i, 0, index_X], edges[i, 0, index_Y],
                        '*', color=c)
                # plot edges
                ax.fill(edges[i, 1:, index_X], edges[i, 1:, index_Y],
                        fill=False)
            # plot spot
            ax.fill(spot[:, index_X], spot[:, index_Y], fill=False)
            ax.fill(spot[:, index_X], spot[:, index_Y], fill=False)


# %%
def define_slits(r0, slit_angles, n_slits, slit_dist, slit_w, slit_l):
    '''
    calculate coordinates of slits edges with central slit at r0
    '''
    n_slits = int(n_slits)
    # calculate slits coordinates:
    r_slits = np.zeros([n_slits, 5, 3])
    for i_slit in range(n_slits):
        # calculate coords of slit center:
        y0 = (n_slits//2 - i_slit)*slit_dist
        r_slits[i_slit, 0, :] = [0., y0, 0.]
        # calculate slit edges:
        r_slits[i_slit, 1, :] = [0., y0 + slit_w/2, slit_l/2]
        r_slits[i_slit, 2, :] = [0., y0 - slit_w/2, slit_l/2]
        r_slits[i_slit, 3, :] = [0., y0 - slit_w/2, -slit_l/2]
        r_slits[i_slit, 4, :] = [0., y0 + slit_w/2, -slit_l/2]
        # rotate and shift to slit position:
        for j in range(5):
            r_slits[i_slit, j, :] = rotate3(r_slits[i_slit, j, :],
                                            slit_angles, slit_angles)
            r_slits[i_slit, j, :] += r0

    # calculate normal to slit plane:
    slit_plane_n = calc_normal(r_slits[0, 0, :], r_slits[0, 1, :],
                               r_slits[0, 2, :])

    # create polygon, which contains all slits (slits spot):
    slits_spot = 1.5*np.vstack([r_slits[0, [1, 4], :] - r0,
                                r_slits[-1, [3, 2], :] - r0]) + r0

    return r_slits, slit_plane_n, slits_spot


# %%
def calc_normal(point1, point2, point3):
    '''
    calculate vector normal to a plane defined by 3 points
    '''
    plane_n = np.cross(point1 - point2, point1 - point3)
    return plane_n/np.linalg.norm(plane_n)


# %% define class for geometry
class Geometry():
    '''
    object containing geometry points
    '''

    def __init__(self):
        # dictionary for Plates objects:
        self.plates_dict = dict()
        # dictionary for positions of all objects:
        self.r_dict = dict()
        # arrays for primary and secondary beamline angles:
        self.angles_dict = dict()
        # determining chamber entrance and exit:
        self.chamb_ent = []
        self.chamb_ext = []
        # dictionary for poloidal field coils:
        self.pf_coils = dict()
        # Tor Field coil contour:
        self.coil = np.array([])
        # vacuum vessel contour:
        self.camera = np.array([])
        # separatrix contour:
        self.sep = np.array([])
        # inner and outer first wall contours:
        self.in_fw = np.array([])
        self.out_fw = np.array([])
        # plasma geometry
        self.R = 0
        self.r_plasma = 0
        self.elon = 0

    def check_chamb_intersect(self, beamline, point1, point2):
        '''
        check intersection between segment 1->2 and chamber
        '''
        intersect_flag = False
        # do not check intersection when particle is far from chamber
        if (point1[0] > 2.5 and point2[1] > 1.5) or \
           (point1[0] < 2.0 and point2[1] < 0.8):
               return intersect_flag
        if beamline == 'prim':
            # check intersection with chamber entrance and chamber at HFS
            # if len(self.chamb_ent) == 0: return False
            for i in np.arange(0, len(self.chamb_ent), 2):
                intersect_flag = intersect_flag or \
                    is_intersect(point1[0:2], point2[0:2],
                                   self.chamb_ent[i], self.chamb_ent[i+1])
        elif beamline == 'sec':
            # check intersection with chamber exit
            # if len(self.chamb_ext) == 0: return False
            for i in np.arange(0, len(self.chamb_ext), 2):
                intersect_flag = intersect_flag or \
                    is_intersect(point1[0:2], point2[0:2],
                                   self.chamb_ext[i], self.chamb_ext[i+1])
        return intersect_flag

    def check_plates_intersect(self, point1, point2):
        '''
        check intersection between segment 1->2 and plates
        '''
        # do not check intersection when particle is outside beamlines
        if point2[0] < self.r_dict['aim'][0]-0.05 and \
           point1[1] < self.r_dict['port_in'][1]:
            return False, 'none'
        for key in self.plates_dict.keys():
            # check if a point in inside the beamline
            if (key in ['A1', 'B1', 'A2', 'B2'] and
                point1[1] > self.r_dict['port_in'][1]) or \
                (key in ['A3', 'A3d', 'B3', 'A4', 'A4d', 'B4'] and
                 point2[0] > self.r_dict['aim'][0]-0.05):
                # check intersection
                if self.plates_dict[key].check_intersect(point1, point2):
                    return True, key
            else:
                continue
        return False, 'none'

    def add_coords(self, name, ref_point, dist, angles):
        '''
        add new element 'name' to r_dict
        '''
        # unpack ref_point
        if type(ref_point) == str:
            r0 = self.r_dict[ref_point]
        else:
            r0 = ref_point
        # unpack angles
        alpha, beta = angles[0:2]
        # coordinates of the center of the object
        r = r0 + calc_vector(dist, alpha, beta)
        self.r_dict[name] = r

    def plot(self, ax, axes='XY', plot_sep=True, plot_aim=True,
             plot_analyzer=True):
        '''
        plot all geometry objects
        '''
        # plot camera and separatrix in XY plane
        if axes == 'XY':
            # plot toroidal coil
            ax.plot(self.coil[:, 0], self.coil[:, 1], '--', color='k')
            ax.plot(self.coil[:, 2], self.coil[:, 3], '--', color='k')
            # plot tokamak camera
            ax.plot(self.camera[:, 0], self.camera[:, 1],
                    color='tab:blue')
            # plot first wall
            ax.plot(self.in_fw[:, 0], self.in_fw[:, 1], color='k')
            ax.plot(self.out_fw[:, 0], self.out_fw[:, 1], color='k')
            # plot separatrix
            if plot_sep:
                ax.plot(self.sep[:, 0] + self.R, self.sep[:, 1],
                        markersize=2, color='b')  # 'tab:orange')
            # plot PF coils
            for coil in self.pf_coils.keys():
                xc = self.pf_coils[coil][0]
                yc = self.pf_coils[coil][1]
                dx = self.pf_coils[coil][2]
                dy = self.pf_coils[coil][3]
                ax.add_patch(Rectangle((xc-dx/2, yc-dy/2), dx, dy,
                                       linewidth=1, edgecolor='tab:gray',
                                       facecolor='tab:gray'))

        index_X, index_Y = get_index(axes)
        # plot plates
        for name in self.plates_dict.keys():
            if name == 'an' and not plot_analyzer:
                continue  # do not plot Analyzer
            self.plates_dict[name].plot(ax, axes=axes)
        if plot_aim:
            # plot aim dot
            ax.plot(self.r_dict['aim'][index_X], self.r_dict['aim'][index_Y],
                    '*', color='b')
            # plot the center of the central slit
            ax.plot(self.r_dict['slit'][index_X], self.r_dict['slit'][index_Y],
                    '*', color='g')


# %%
def add_diafragm(geom, plts_name, diaf_name, diaf_width=0.1):
    '''
    add new plates object which works as a diafragm
    '''
    # create new object in plates dictionary as a copy of existing plates
    geom.plates_dict[diaf_name] = copy.deepcopy(geom.plates_dict[plts_name])
    angles = geom.plates_dict[diaf_name].angles
    beamline_angles = geom.plates_dict[diaf_name].beamline_angles
    r0 = geom.r_dict[plts_name]
    for i in [0, 1]:  # index for upper/lower plate
        for j in [0, 1]:
            # rotate and shift edge to initial coord system
            coords = rotate3(geom.plates_dict[diaf_name].edges[i][j] -
                             r0, angles, beamline_angles, inverse=True)
            # shift up for upper plate and down for lower
            coords += [0, diaf_width*(1-2*i), 0]
            geom.plates_dict[diaf_name].edges[i][3-j] = \
                rotate3(coords, angles, beamline_angles, inverse=False) + r0


# %%
@numba.njit()
def calc_vector(length, alpha, beta, direction=(1, 1, -1)):
    '''
    calculate vector based on its length and angles
    '''
    drad = np.pi/180.  # converts degrees to radians
    x = direction[0] * length * np.cos(alpha*drad) * np.cos(beta*drad)
    y = direction[1] * length * np.sin(alpha*drad)
    z = direction[2] * length * np.cos(alpha*drad) * np.sin(beta*drad)
    return np.array([x, y, z])


@numba.njit()
def calc_angles(vector):
    '''
    calculate alpha and beta angles based on vector coords
    '''
    drad = np.pi/180.  # converts degrees to radians
    x, y, z = vector / np.linalg.norm(vector)
    # alpha = np.arcsin(y)  # rad
    alpha = np.arccos(x)  # rad
    if abs(y) > 1e-9:
        beta = np.arcsin(-np.tan(alpha) * z / y)  # rad
    elif abs(z) < 1e-9:
        beta = 0.
    elif abs(x) > 1e-9:
        beta = np.arctan(-z / x)  # rad
    else:
        beta = -np.sign(z) * np.pi/2
    return alpha/drad, beta/drad  # degrees


# %% get axes index
def get_index(axes):
    axes_dict = {'XY': (0, 1), 'XZ': (0, 2), 'ZY': (2, 1)}
    return axes_dict[axes]


# %% Runge-Kutta
# define equations of movement:

@numba.njit()
def f(k, E, V, B):
    return k*(E + np.cross(V, B))


@numba.njit()
def g(V):
    return V


@numba.njit()
def runge_kutt(k, RV, dt, E, B):
    '''
    Calculate one step using Runge-Kutta algorithm

    V' = k(E + [VxB]) == K(E + np.cross(V,B)) == f
    r' = V == g

    V[n+1] = V[n] + (h/6)(m1 + 2m2 + 2m3 + m4)
    r[n+1] = r[n] + (h/6)(k1 + 2k2 + 2k3 + k4)
    m[1] = f(t[n], V[n], r[n])
    k[1] = g(t[n], V[n], r[n])
    m[2] = f(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    k[2] = g(t[n] + (h/2), V[n] + (h/2)m[1], r[n] + (h/2)k[1])
    m[3] = f(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    k[3] = g(t[n] + (h/2), V[n] + (h/2)m[2], r[n] + (h/2)k[2])
    m[4] = f(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])
    k[4] = g(t[n] + h, V[n] + h*m[3], r[n] + h*k[3])

    Parameters
    ----------
    k : float
        particle charge [Co] / particle mass [kg]
    RV : np.array([[x, y, z, Vx, Vy, Vz]])
        coordinates and velocities array [m], [m/s]
    dt : float
        timestep [s]
    E : np.array([Ex, Ey, Ez])
        values of electric field at current point [V/m]
    B : np.array([Bx, By, Bz])
        values of magnetic field at current point [T]

    Returns
    -------
    RV : np.array([[x, y, z, Vx, Vy, Vz]])
        new coordinates and velocities

    '''
    r = RV[0, :3]
    V = RV[0, 3:]

    m1 = f(k, E, V, B)
    k1 = g(V)

    fV2 = V + (dt / 2.) * m1
    gV2 = V + (dt / 2.) * m1
    m2 = f(k, E, fV2, B)
    k2 = g(gV2)

    fV3 = V + (dt / 2.) * m2
    gV3 = V + (dt / 2.) * m2
    m3 = f(k, E, fV3, B)
    k3 = g(gV3)

    fV4 = V + dt * m3
    gV4 = V + dt * m3
    m4 = f(k, E, fV4, B)
    k4 = g(gV4)

    V = V + (dt / 6.) * (m1 + (2. * m2) + (2. * m3) + m4)
    r = r + (dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4)

    RV = np.hstack((r, V))
    return RV


# %%
def optimize_B2(tr, geom, UB2, dUB2, E, B, dt, stop_plane_n, target='aim',
                optimize=True, eps_xy=1e-3, eps_z=1e-3, dt_min=1e-10):
    '''
    get voltages on B2 plates and choose secondary trajectory
    which goes into target
    '''
    # set up target
    print('Target: ' + target)
    r_aim = geom.r_dict[target]
    attempts_opt = 0
    attempts_fan = 0
    while True:
        tr.U['B2'], tr.dt1, tr.dt2 = UB2, dt, dt
        # pass fan of secondaries
        tr.pass_fan(r_aim, E, B, geom, stop_plane_n=stop_plane_n,
                    eps_xy=eps_xy, eps_z=eps_z,
                    no_intersect=True, no_out_of_bounds=True)
        # pass trajectory to the target
        tr.pass_to_target(r_aim, E, B, geom, stop_plane_n=stop_plane_n,
                          eps_xy=eps_xy, eps_z=eps_z, dt_min=dt_min,
                          no_intersect=True, no_out_of_bounds=True)
        print('IsAimXY = ', tr.IsAimXY)
        print('IsAimZ = ', tr.IsAimZ)
        if True in tr.IntersectGeometry.values():
            break
        if not tr.fan_ok:
            attempts_fan += 1
        if attempts_fan > 3 or len(tr.Fan) == 0:
            print('Fan of secondaries is not ok')
            break

        if optimize:
            # change UB2 value proportional to dz
            if not tr.IsAimZ:
                dz = r_aim[2]-tr.RV_sec[-1, 2]
                print('UB2 OLD = {:.2f}, z_aim - z = {:.4f} m'
                      .format(UB2, dz))
                UB2 = UB2 - dUB2*dz
                print('UB2 NEW = {:.2f}'.format(UB2))
                attempts_opt += 1
            else:
                break
            # check if there is a loop while finding secondary to aim
            if attempts_opt > 20:
                print('too many attempts B2!')
                break
        else:
            print('B2 was not optimized')
            break
    return tr


# %%
def optimize_A3B3(tr, geom, UA3, UB3, dUA3, dUB3,
                  E, B, dt, target='slit', UA3_max=50., UB3_max=50.,
                  eps_xy=1e-3, eps_z=1e-3):
    '''
    get voltages on A3 and B3 plates to get into target
    '''
    print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U['A2']))
    print('Target: ' + target)
    if target == 'slit':
        rs = geom.r_dict['slit']
        stop_plane_n = geom.plates_dict['an'].slit_plane_n
    elif target == 'det':
        rs = geom.r_dict['det']
        stop_plane_n = geom.plates_dict['an'].det_plane_n
    elif target == 'A4':
        rs = geom.r_dict['A4']
        stop_plane_n = geom.plates_dict['an'].slit_plane_n

    tr.dt1 = dt
    tr.dt2 = dt
    tmax = 9e-5
    tr.IsAimXY = False
    tr.IsAimZ = False
    RV0 = np.array([tr.RV_sec[0]])

    vltg_fail = False  # flag to track voltage failure
    n_stepsA3 = 0
    while not (tr.IsAimXY and tr.IsAimZ):
        tr.U['A3'], tr.U['B3'] = UA3, UB3
        tr.pass_sec(RV0, rs, E, B, geom,  # stop_plane_n=stop_plane_n,
                    tmax=tmax, eps_xy=eps_xy, eps_z=eps_z)

        drXY = np.linalg.norm(rs[:2]-tr.RV_sec[-1, :2]) * \
            np.sign(np.cross(tr.RV_sec[-1, :2], rs[:2]))
        print('\n UA3 OLD = {:.2f} kV, dr XY = {:.4f} m'.format(UA3, drXY))
        print('IsAimXY = ', tr.IsAimXY)
        # if drXY < 1e-2:
        #     dUA3 = 10.0

        UA3 = UA3 + dUA3*drXY
        print('UA3 NEW = {:.2f} kV'.format(UA3))
        n_stepsA3 += 1

        if abs(UA3) > UA3_max:
            print('ALPHA3 failed, voltage too high')
            vltg_fail = True
            return tr, vltg_fail
        if n_stepsA3 > 100:
            print('ALPHA3 failed, too many steps')
            vltg_fail = True
            return tr, vltg_fail

        # dz = rs[2] - tr.RV_sec[-1, 2]
        # print('\n UB3 OLD = {:.2f} kV, dZ = {:.4f} m'.format(UB3, dz))
        if abs(drXY) < 0.01:
            if tr.IntersectGeometrySec['A3']:
                print('BAD A3!')
                vltg_fail = True
                return tr, vltg_fail
            n_stepsZ = 0
            while not tr.IsAimZ:
                print('pushing Z direction')
                tr.U['A3'], tr.U['B3'] = UA3, UB3
                tr.pass_sec(RV0, rs, E, B, geom,  # stop_plane_n=stop_plane_n,
                            tmax=tmax, eps_xy=eps_xy, eps_z=eps_z)
                # tr.IsAimZ = True  # if you want to skip UB3 calculation
                dz = rs[2] - tr.RV_sec[-1, 2]
                print(' UB3 OLD = {:.2f} kV, dZ = {:.4f} m'
                      .format(UB3, dz))
                print('IsAimXY = ', tr.IsAimXY)
                print('IsAimZ = ', tr.IsAimZ)

                UB3 = UB3 - dUB3*dz
                n_stepsZ += 1
                if abs(UB3) > UB3_max:
                    print('BETA3 failed, voltage too high')
                    vltg_fail = True
                    return tr, vltg_fail
                if n_stepsZ > 100:
                    print('BETA3 failed, too many steps')
                    vltg_fail = True
                    return tr, vltg_fail
                # print('UB3 NEW = {:.2f} kV'.format(UB3))
            n_stepsA3 = 0
            print('n_stepsZ = ', n_stepsZ)
            dz = rs[2] - tr.RV_sec[-1, 2]
            print('UB3 NEW = {:.2f} kV, dZ = {:.4f} m'.format(UB3, dz))

    return tr, vltg_fail


# %%
def optimize_A4(tr, geom, UA4, dUA4, E, B, dt, eps_alpha=0.1):
    '''
    get voltages on A4 to get proper alpha angle at the entrance to analyzer
    '''
    print('\n A4 optimization\n')
    print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U['A2']))

    rs = geom.r_dict['slit']
    stop_plane_n = geom.plates_dict['an'].slit_plane_n
    alpha_target = geom.angles_dict['an']

    tr.dt1 = dt
    tr.dt2 = dt
    tmax = 9e-5
    # tr.IsAimXY = False
    # tr.IsAimZ = False
    RV0 = np.array([tr.RV_sec[0]])
    V_last = tr.RV_sec[-1][3:]
    alpha, beta = calc_angles(V_last)
    dalpha = alpha_target - alpha
    n_stepsA4 = 0
    while (abs(alpha - alpha_target) > eps_alpha):
        tr.U['A4'] = UA4
        tr.pass_sec(RV0, rs, E, B, geom,  # stop_plane_n=stop_plane_n,
                    tmax=tmax, eps_xy=1e-2, eps_z=1e-2)

        V_last = tr.RV_sec[-1][3:]
        alpha, beta = calc_angles(V_last)
        dalpha = alpha_target - alpha
        print('\n UA4 OLD = {:.2f} kV, dalpha = {:.4f} deg'
              .format(UA4, dalpha))
        drXY = np.linalg.norm(rs[:2]-tr.RV_sec[-1, :2]) * \
            np.sign(np.cross(tr.RV_sec[-1, :2], rs[:2]))
        dz = rs[2] - tr.RV_sec[-1, 2]
        print('dr XY = {:.4f} m, dz = {:.4f} m'.format(drXY, dz))

        UA4 = UA4 + dUA4*dalpha
        print('UA4 NEW = {:.2f} kV'.format(UA4))
        n_stepsA4 += 1

        if abs(UA4) > 50.:
            print('ALPHA4 failed, voltage too high')
            return tr
        if n_stepsA4 > 100:
            print('ALPHA4 failed, too many steps')
            return tr

    return tr


# %%
def calc_zones(tr, dt, E, B, geom, slits=[2], timestep_divider=5,
               stop_plane_n=np.array([1, 0, 0]), eps_xy=1e-3, eps_z=1,
               dt_min=1e-11, no_intersect=True, no_out_of_bounds=True):
    '''
    calculate ionization zones
    '''
    # find the number of slits
    n_slits = geom.plates_dict['an'].slits_edges.shape[0]
    tr.add_slits(n_slits)
    # set target at the central slit
    r_aim = geom.plates_dict['an'].slits_edges[n_slits//2, 0, :]

    # create slits polygon
    slit_plane_n = geom.plates_dict['an'].slit_plane_n
    slits_spot = geom.plates_dict['an'].slits_spot
    ax_index = np.argmax(slit_plane_n)
    slits_spot_flat = np.delete(slits_spot, ax_index, 1)
    slits_spot_poly = path.Path(slits_spot_flat)

    # find trajectories which go to upper and lower slit edge
    # find index of primary trajectory point where secondary starts
    index = np.nanargmin(np.linalg.norm(tr.RV_prim[:, :3] -
                                        tr.RV_sec[0, :3], axis=1))
    # set up the index range
    sec_ind = range(index-2, index+2)
    print('\nStarting precise fan calculation')
    k = tr.q / tr.m
    # divide the timestep
    tr.dt1 = dt/timestep_divider
    tr.dt2 = dt
    # number of steps during new fan calculation
    n_steps = timestep_divider * (len(sec_ind))
    # list for new trajectories
    fan_list = []
    # take the point to start fan calculation
    # RV_old = tr.Fan[sec_ind[0]-1][0]
    RV_old = tr.RV_prim[sec_ind[0]]
    RV_old = np.array([RV_old])
    RV_new = RV_old

    i_steps = 0
    while i_steps <= n_steps:
        # pass new secondary trajectory
        tr.pass_sec(RV_new, r_aim, E, B, geom,
                    stop_plane_n=slit_plane_n,
                    tmax=9e-5, eps_xy=1e-3, eps_z=1)
        # make a step on primary trajectory
        r = RV_old[0, :3]
        B_local = return_B(r, B)
        E_local = np.array([0., 0., 0.])
        RV_new = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
        RV_old = RV_new
        i_steps += 1
        # check intersection with slits polygon
        intersect_coords_flat = np.delete(tr.RV_sec[-1, :3], ax_index, 0)
        contains_point = slits_spot_poly.contains_point(intersect_coords_flat)
        if not (True in tr.IntersectGeometrySec.values() or
                tr.B_out_of_bounds) and contains_point:
            fan_list.append(tr.RV_sec)
    tr.Fan = fan_list
    tr.fan_to_slits = fan_list
    print('\nPrecise fan calculated')

    for i_slit in slits:
        # set up upper and lower slit edge
        upper_edge = [geom.plates_dict['an'].slits_edges[i_slit, 4, :],
                      geom.plates_dict['an'].slits_edges[i_slit, 1, :]]
        lower_edge = [geom.plates_dict['an'].slits_edges[i_slit, 3, :],
                      geom.plates_dict['an'].slits_edges[i_slit, 2, :]]
        zones_list = []  # list for ion zones coordinates
        rv_list = []  # list for RV arrays of secondaries
        for edge in [upper_edge, lower_edge]:
            # find intersection of fan and slit edge
            for i_tr in range(len(tr.Fan) - 1):
                p1, p2 = tr.Fan[i_tr][-1, :3], tr.Fan[i_tr+1][-1, :3]
                # check intersection between fan segment and slit edge
                if is_intersect(p1, p2, edge[0], edge[1]):
                    r_intersect = segm_intersect(p1, p2, edge[0], edge[1])
                    print('\n intersection with slit ' + str(i_slit))
                    print(r_intersect)
                    tr.dt1 = dt/timestep_divider
                    tr.dt2 = dt
                    tr.pass_to_target(r_intersect, E, B, geom,
                                      eps_xy=eps_xy, eps_z=eps_z,
                                      dt_min=dt_min,
                                      stop_plane_n=slit_plane_n,
                                      no_intersect=no_intersect,
                                      no_out_of_bounds=no_out_of_bounds)
                    zones_list.append(tr.RV_sec[-1, :3])
                    rv_list.append(tr.RV_sec)
                    print('ok!')
                    break
        tr.ion_zones[i_slit] = np.array(zones_list)
        tr.RV_sec_toslits[i_slit] = rv_list
    return tr


# %%
def pass_to_slits(tr, dt, E, B, geom, target='slit', timestep_divider=10,
                   slits=range(5), no_intersect=True, no_out_of_bounds=True):
    '''
    pass trajectories to slits and save secondaries which get into slits
    '''
    tr.dt1 = dt/4
    tr.dt2 = dt
    k = tr.q / tr.m
    # find the number of slits
    n_slits = geom.plates_dict['an'].slits_edges.shape[0]
    tr.add_slits(n_slits)
    # find slits position
    if target == 'slit':
        r_slits = geom.plates_dict['an'].slits_edges
        rs = geom.r_dict['slit']
        slit_plane_n = geom.plates_dict['an'].slit_plane_n
        slits_spot = geom.plates_dict['an'].slits_spot
    elif target == 'det':
        r_slits = geom.plates_dict['an'].det_edges
        rs = geom.r_dict['det']
        slit_plane_n = geom.plates_dict['an'].det_plane_n
        slits_spot = geom.plates_dict['an'].det_spot

    # create slits polygon
    ax_index = np.argmax(slit_plane_n)
    slits_spot_flat = np.delete(slits_spot, ax_index, 1)
    slits_spot_poly = path.Path(slits_spot_flat)

    # find index of primary trajectory point where secondary starts
    index = np.nanargmin(np.linalg.norm(tr.RV_prim[:, :3] -
                                        tr.RV_sec[0, :3], axis=1))
    sec_ind = range(index-2, index+2)

    print('\nStarting precise fan calculation')
    # divide the timestep
    tr.dt1 = dt/timestep_divider
    tr.dt2 = dt
    # number of steps during new fan calculation
    n_steps = timestep_divider * (len(sec_ind))
    # list for new trajectories
    fan_list = []
    # take the point to start fan calculation
    # RV_old = tr.Fan[sec_ind[0]-1][0]
    RV_old = tr.RV_prim[sec_ind[0]]
    RV_old = np.array([RV_old])
    RV_new = RV_old

    i_steps = 0
    inside_slits_poly = False
    while i_steps <= n_steps:
        # pass new secondary trajectory
        tr.pass_sec(RV_new, rs, E, B, geom,
                    stop_plane_n=slit_plane_n, tmax=9e-5,
                    eps_xy=1e-3, eps_z=1)
        # make a step on primary trajectory
        r = RV_old[0, :3]
        B_local = return_B(r, B)
        E_local = np.array([0., 0., 0.])
        RV_new = runge_kutt(k, RV_old, tr.dt1, E_local, B_local)
        RV_old = RV_new
        i_steps += 1
        intersect_coords_flat = np.delete(tr.RV_sec[-1, :3], ax_index, 0)
        contains_point = slits_spot_poly.contains_point(intersect_coords_flat)
        if not (True in tr.IntersectGeometrySec.values() or
                tr.B_out_of_bounds) and contains_point:
            inside_slits_poly = True
            fan_list.append(tr.RV_sec)
        if not contains_point and inside_slits_poly:
            break
    print('\nPrecise fan calculated')

    # choose secondaries which get into slits
    # start slit cycle
    for i_slit in slits:
        print('\nslit = {}'.format(i_slit+1))
        print('center of the slit = ', r_slits[i_slit, 0, :], '\n')
        # create slit polygon
        slit_flat = np.delete(r_slits[i_slit, 1:, :], ax_index, 1)
        slit_poly = path.Path(slit_flat)
        zones_list = []  # list for ion zones coordinates
        rv_list = []  # list for RV arrays of secondaries
        for fan_tr in fan_list:
            # get last coordinates of the secondary trajectory
            intersect_coords_flat = np.delete(fan_tr[-1, :3], ax_index, 0)
            if slit_poly.contains_point(intersect_coords_flat):
                print('slit {} ok!\n'.format(i_slit+1))
                rv_list.append(fan_tr)
                zones_list.append(fan_tr[0, :3])

        tr.RV_sec_toslits[i_slit] = rv_list
        tr.ion_zones[i_slit] = np.array(zones_list)
    tr.fan_to_slits = fan_list

    return tr


# %%
@numba.njit()
def translate(input_array, xyz):
    '''
    move the vector in space
    xyz : 3 component vector
    '''
    if input_array is not None:
        input_array += np.array(xyz)

    return input_array


@numba.njit()
def rot_mx(axis=(1, 0, 0), deg=0):
    '''
    function calculates rotation matrix
    '''
    n = axis
    ca = np.cos(np.radians(deg))
    sa = np.sin(np.radians(deg))
    R = np.array([[n[0]**2*(1-ca)+ca, n[0]*n[1]*(1-ca)-n[2]*sa,
                   n[0]*n[2]*(1-ca)+n[1]*sa],

                  [n[1]*n[0]*(1-ca)+n[2]*sa, n[1]**2*(1-ca)+ca,
                   n[1]*n[2]*(1-ca)-n[0]*sa],

                  [n[2]*n[0]*(1-ca)-n[1]*sa, n[2]*n[1]*(1-ca)+n[0]*sa,
                   n[2]**2*(1-ca)+ca]])
    return R


@numba.njit()
def rotate(input_array, axis=(1, 0, 0), deg=0.):
    '''
    rotate vector around given axis by deg [degrees]
    axis : axis of rotation
    deg : angle in degrees
    '''
    if input_array is not None:
        R = rot_mx(axis, deg)
        input_array = np.dot(input_array, R.T)
    return input_array


@numba.njit()
def rotate3(input_array, plates_angles, beamline_angles, inverse=False):
    '''
    rotate vector in 3 dimentions
    plates_angles : angles of the plates
    beamline_angles : angles of the beamline axis, rotation on gamma angle
    '''
    alpha, beta, gamma = plates_angles
    axis = calc_vector(1, beamline_angles[0], beamline_angles[1])

    if inverse:
        rotated_array = rotate(input_array, axis=axis, deg=-gamma)
        rotated_array = rotate(rotated_array, axis=(0, 1, 0), deg=-beta)
        rotated_array = rotate(rotated_array, axis=(0, 0, 1), deg=-alpha)
    else:
        rotated_array = rotate(input_array, axis=(0, 0, 1), deg=alpha)
        rotated_array = rotate(rotated_array, axis=(0, 1, 0), deg=beta)
        rotated_array = rotate(rotated_array, axis=axis, deg=gamma)
    return rotated_array


# %% Intersection check functions
@numba.njit()
def line_plane_intersect(planeNormal, planePoint, rayDirection,
                         rayPoint, eps=1e-6):
    '''
    function returns intersection point between plane and ray
    '''
    ndotu = np.dot(planeNormal, rayDirection)
    if abs(ndotu) < eps:
        # print('no intersection or line is within plane')
        return np.full_like(planeNormal, np.nan)
    else:
        w = rayPoint - planePoint
        si = -np.dot(planeNormal, w) / ndotu
        Psi = w + si * rayDirection + planePoint
        return Psi


@numba.njit()
def is_between(A, B, C, eps=1e-6):
    '''
    function returns True if point C is on the segment AB (between A and B)
    '''
    if np.isnan(C).any():
        return False
    # check if the points are on the same line
    crossprod = np.cross(B-A, C-A)
    if np.linalg.norm(crossprod) > eps:
        return False
    # check if the point is between
    dotprod = np.dot(B-A, C-A)
    if dotprod < 0 or dotprod > np.linalg.norm(B-A)**2:
        return False
    return True


@numba.njit()
def order(A, B, C):
    '''
    if counterclockwise return True
    if clockwise return False
    '''
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


@numba.njit()
def is_intersect(A, B, C, D):  # doesn't work with collinear case
    '''
    function returns true if line segments AB and CD intersect
    '''
    # Return true if line segments AB and CD intersect
    return order(A, C, D) != order(B, C, D) and \
        order(A, B, C) != order(A, B, D)


@numba.njit()
def segm_intersect(A, B, C, D):
    '''
    function calculates intersection point between vectors AB and CD
    in case AB and CD intersect
    '''
    # define vectors
    AB, CA, CD = B - A, A - C, D - C
    return A + AB * (np.cross(CD, CA) / np.cross(AB, CD))


@numba.njit()
def segm_poly_intersect(polygon_coords, segment_coords):
    '''
    check segment and polygon intersection
    '''
    polygon_normal = np.cross(polygon_coords[2, 0:3]-polygon_coords[0, 0:3],
                              polygon_coords[1, 0:3]-polygon_coords[0, 0:3])
    polygon_normal = polygon_normal/np.linalg.norm(polygon_normal)
    # find the intersection point between polygon plane and segment line
    intersect_coords = line_plane_intersect(polygon_normal,
                                            polygon_coords[2, 0:3],
                                            segment_coords[1, 0:3] -
                                            segment_coords[0, 0:3],
                                            segment_coords[0, 0:3])
    if np.isnan(intersect_coords).any():
        return False
    if not is_between(segment_coords[0, 0:3], segment_coords[1, 0:3],
                      intersect_coords):
        return False
    # go to 2D, exclude the maximum coordinate
    i = np.argmax(np.abs(polygon_normal))
    inds = np.array([0, 1, 2])  # indexes for 3d corrds
    inds_flat = np.where(inds != i)[0]
    polygon_coords_flat = polygon_coords[:, inds_flat]
    intersect_coords_flat = intersect_coords[inds_flat]
    # define a rectange which contains the flat poly
    xmin = np.min(polygon_coords_flat[:, 0])
    xmax = np.max(polygon_coords_flat[:, 0])
    ymin = np.min(polygon_coords_flat[:, 1])
    ymax = np.max(polygon_coords_flat[:, 1])
    xi, yi = intersect_coords_flat
    # simple check if a point is inside a rectangle
    if (xi < xmin or xi > xmax or yi < ymin or yi > ymax):
        return False
    # ray casting algorithm
    # set up a point outside the flat poly
    point_out = np.array([xmin - 0.01, ymin - 0.01])
    # calculate the number of intersections between ray and the poly sides
    intersections = 0
    for i in range(polygon_coords_flat.shape[0]):
        if is_intersect(point_out, intersect_coords_flat,
                        polygon_coords_flat[i-1], polygon_coords_flat[i]):
            intersections += 1
    # if the number of intersections is odd then the point is inside
    if intersections % 2 == 0:
        return False
    else:
        return True

    # p = path.Path(polygon_coords_flat)
    # return p.contains_point(intersect_coords_flat)

    # check projections on XY and XZ planes
    # pXY = path.Path(polygon_coords[:, [0, 1]])  # XY plane
    # pXZ = path.Path(polygon_coords[:, [0, 2]])  # XZ plane
    # return pXY.contains_point(intersect_coords[[0, 1]]) and \
    #     pXZ.contains_point(intersect_coords[[0, 2]]) and \
    #         is_between(segment_coords[0, 0:3], segment_coords[1, 0:3],
    #                   intersect_coords)


# %%
def plate_flags(range_x, range_y, range_z, U,
                plts_geom, plts_angles, plts_center):
    '''
    calculate plates cooedinates and boolean arrays for plates
    '''
    length, width, thick, gap, l_sw = plts_geom
    gamma, alpha_sw = plts_angles
    r_sweep_up = np.array([-length/2 + l_sw, gap/2., 0])
    r_sweep_lp = np.array([-length/2 + l_sw, -gap/2., 0])
    # Geometry in system based on central point between plates
    # upper plate
    UP1 = np.array([-length/2., gap/2. + thick, width/2.])
    UP2 = np.array([-length/2., gap/2. + thick, -width/2.])
    UP1sw = np.array([-length/2. + l_sw, gap/2. + thick, width/2.])
    UP2sw = np.array([-length/2. + l_sw, gap/2. + thick, -width/2.])
    UP3 = np.array([length/2., gap/2. + thick, -width/2.])
    UP4 = np.array([length/2., gap/2. + thick, width/2.])
    UP5 = np.array([-length/2., gap/2., width/2.])
    UP6 = np.array([-length/2., gap/2., -width/2.])
    UP5sw = np.array([-length/2. + l_sw, gap/2., width/2.])
    UP6sw = np.array([-length/2. + l_sw, gap/2., -width/2.])
    UP7 = np.array([length/2., gap/2., -width/2.])
    UP8 = np.array([length/2., gap/2., width/2.])
    if abs(alpha_sw) > 1e-2:
        UP1 = UP1sw + rotate(UP1 - UP1sw, axis=(0, 0, 1), deg=-alpha_sw)
        UP2 = UP2sw + rotate(UP2 - UP2sw, axis=(0, 0, 1), deg=-alpha_sw)
        UP5 = UP5sw + rotate(UP5 - UP5sw, axis=(0, 0, 1), deg=-alpha_sw)
        UP6 = UP6sw + rotate(UP6 - UP6sw, axis=(0, 0, 1), deg=-alpha_sw)
        # points are sorted clockwise
        UP_full = np.array([UP1sw, UP1, UP2, UP2sw, UP3, UP4,
                            UP5sw, UP5, UP6, UP6sw, UP7, UP8])
    else:
        UP_full = np.array([UP1, UP2, UP3, UP4, UP5, UP6, UP7, UP8])
    UP_rotated = UP_full.copy()
    for i in range(UP_full.shape[0]):
        UP_rotated[i, :] = rotate(UP_rotated[i, :], axis=(1, 0, 0), deg=gamma)
        # shift coords center
        UP_rotated[i, :] += plts_center

    # lower plate
    LP1 = np.array([-length/2., -gap/2. - thick, width/2.])
    LP2 = np.array([-length/2., -gap/2. - thick, -width/2.])
    LP1sw = np.array([-length/2. + l_sw, -gap/2. - thick, width/2.])
    LP2sw = np.array([-length/2. + l_sw, -gap/2. - thick, -width/2.])
    LP3 = np.array([length/2., -gap/2. - thick, -width/2.])
    LP4 = np.array([length/2., -gap/2. - thick, width/2.])
    LP5 = np.array([-length/2., -gap/2., width/2.])
    LP6 = np.array([-length/2., -gap/2., -width/2.])
    LP5sw = np.array([-length/2. + l_sw, -gap/2., width/2.])
    LP6sw = np.array([-length/2. + l_sw, -gap/2., -width/2.])
    LP7 = np.array([length/2., -gap/2., -width/2.])
    LP8 = np.array([length/2., -gap/2., width/2.])
    if abs(alpha_sw) > 1e-2:
        LP1 = LP1sw + rotate(LP1 - LP1sw, axis=(0, 0, 1), deg=alpha_sw)
        LP2 = LP2sw + rotate(LP2 - LP2sw, axis=(0, 0, 1), deg=alpha_sw)
        LP5 = LP5sw + rotate(LP5 - LP5sw, axis=(0, 0, 1), deg=alpha_sw)
        LP6 = LP6sw + rotate(LP6 - LP6sw, axis=(0, 0, 1), deg=alpha_sw)
        # points are sorted clockwise
        LP_full = np.array([LP1sw, LP1, LP2, LP2sw, LP3, LP4,
                            LP5sw, LP5, LP6, LP6sw, LP7, LP8])
    else:
        LP_full = np.array([LP1, LP2, LP3, LP4, LP5, LP6, LP7, LP8])
    LP_rotated = LP_full.copy()
    for i in range(LP_full.shape[0]):
        LP_rotated[i, :] = rotate(LP_rotated[i, :], axis=(1, 0, 0), deg=gamma)
        # shift coords center
        LP_rotated[i, :] += plts_center

    # Find coords of 'cubes' containing each plate
    upper_cube = np.array([np.min(UP_rotated, axis=0),
                           np.max(UP_rotated, axis=0)])
    lower_cube = np.array([np.min(LP_rotated, axis=0),
                           np.max(LP_rotated, axis=0)])

    # create mask for plates
    upper_plate_flag = np.full_like(U, False, dtype=bool)
    lower_plate_flag = np.full_like(U, False, dtype=bool)
    for i in range(range_x.shape[0]):
        for j in range(range_y.shape[0]):
            for k in range(range_z.shape[0]):
                x = range_x[i]
                y = range_y[j]
                z = range_z[k]
                # check upper cube
                if (x >= upper_cube[0, 0]) and (x <= upper_cube[1, 0]) and \
                   (y >= upper_cube[0, 1]) and (y <= upper_cube[1, 1]) and \
                   (z >= upper_cube[0, 2]) and (z <= upper_cube[1, 2]):
                    r = np.array([x, y, z]) - plts_center
                    # inverse rotation
                    r_rot = rotate(r, axis=(1, 0, 0), deg=-gamma)
                    if r_rot[0] <= -length/2 + l_sw:
                        r_rot = r_sweep_up + rotate(r_rot - r_sweep_up,
                                                 axis=(0, 0, 1), deg=alpha_sw)
                    # define masks for upper and lower plates
                    upper_plate_flag[i, j, k] = (r_rot[0] >= -length/2.) and \
                        (r_rot[0] <= length/2.) and (r_rot[2] >= -width/2.) and \
                        (r_rot[2] <= width/2.) and (r_rot[1] >= gap/2.) and \
                        (r_rot[1] <= gap/2. + thick)
                # check lower cube
                if (x >= lower_cube[0, 0]) and (x <= lower_cube[1, 0]) and \
                   (y >= lower_cube[0, 1]) and (y <= lower_cube[1, 1]) and \
                   (z >= lower_cube[0, 2]) and (z <= lower_cube[1, 2]):
                    r = np.array([x, y, z]) - plts_center
                    # inverse rotation
                    r_rot = rotate(r, axis=(1, 0, 0), deg=-gamma)
                    if r_rot[0] <= -length/2 + l_sw:
                        r_rot = r_sweep_lp + rotate(r_rot - r_sweep_lp,
                                                 axis=(0, 0, 1), deg=-alpha_sw)
                    # define masks for upper and lower plates
                    lower_plate_flag[i, j, k] = (r_rot[0] >= -length/2.) and \
                        (r_rot[0] <= length/2.) and (r_rot[2] >= -width/2.) and \
                        (r_rot[2] <= width/2.) and \
                        (r_rot[1] >= -gap/2. - thick) and \
                        (r_rot[1] <= -gap/2.)

    return UP_rotated, LP_rotated, upper_plate_flag, lower_plate_flag


def return_E(r, Ein, U, geom):
    '''
    take dot and try to interpolate electiric field
    Ein : dict of interpolants for Ex, Ey, Ez
    U : dict with plates voltage values
    '''
    Etotal = np.zeros(3)
    # do not check plates while particle is in plasma
    if r[0] < geom.r_dict['aim'][0]-0.05 and r[1] < geom.r_dict['port_in'][1]:
        return Etotal
    # go through all the plates
    for key in Ein.keys():
        # shift the center of coord system
        r_new = r - geom.r_dict[key]
        # get angles
        angles = copy.deepcopy(geom.plates_dict[key].angles)
        beamline_angles = copy.deepcopy(geom.plates_dict[key].beamline_angles)
        # rotate point to the coord system of plates
        r_new = rotate3(r_new, angles, beamline_angles, inverse=True)
        # interpolate Electric field
        Etemp = np.zeros(3)
        try:
            Etemp[0] = Ein[key][0](r_new) * U[key]
            Etemp[1] = Ein[key][1](r_new) * U[key]
            Etemp[2] = Ein[key][2](r_new) * U[key]
            # rotate Etemp
            Etemp = rotate3(Etemp, angles, beamline_angles, inverse=False)
            # add the result to total E field
            Etotal += Etemp
        except (ValueError, IndexError):
            continue
    return Etotal


def return_B(r, Bin):
    '''
    interpolate Magnetic field at point r
    '''
    Bx_interp, By_interp, Bz_interp = Bin[0], Bin[1], Bin[2]
    Bout = np.c_[Bx_interp(r), By_interp(r), Bz_interp(r)]
    return Bout


def save_E(beamline, plts_name, Ex, Ey, Ez, plts_angles, plts_geom,
           domain, an_params, plate1, plate2, dirname='elecfield'):
    '''
    save Ex, Ey, Ez arrays to file
    '''
    dirname = dirname + '/' + beamline

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o700)
            print("Directory ", dirname, " created ")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    fname = plts_name + '_geometry.dat'
    # erases data from file before writing
    open(dirname + '/' + fname, 'w').close()
    with open(dirname + '/' + fname, 'w') as myfile:
        myfile.write(np.array2string(plts_geom)[1:-1] +
                     ' # plate\'s length, width, thic, gap and l_sweeped\n')
        myfile.write(np.array2string(plts_angles)[1:-1] +
                     ' # plate\'s gamma and alpha_sweep angles\n')
        myfile.write(np.array2string(domain, max_line_width=200)[1:-1] +
                     ' # xmin, xmax, ymin, ymax, zmin, zmax, delta\n')
        if plts_name == 'an':
            myfile.write(np.array2string(an_params, max_line_width=200)[1:-1] +
                         ' # n_slits, slit_dist, slit_w, G, theta, XD, YD1, YD2\n')
        for i in range(plate1.shape[0]):
            myfile.write(np.array2string(plate1[i], precision=4)[1:-1] +
                         ' # 1st plate rotated\n')
        for i in range(plate2.shape[0]):
            myfile.write(np.array2string(plate2[i], precision=4)[1:-1] +
                         ' # 2nd plate rotated\n')

    np.save(dirname + '/' + plts_name + '_Ex', Ex)
    np.save(dirname + '/' + plts_name + '_Ey', Ey)
    np.save(dirname + '/' + plts_name + '_Ez', Ez)

    print('Electric field saved, ' + plts_name + '\n')


def read_plates(beamline, geom, E, dirname='elecfield'):
    '''
    read Electric field and plates geometry
    '''
    dirname = dirname + '/' + beamline

    # list of all *.dat files
    file_list = [file for file in os.listdir(dirname) if file.endswith('dat')]
    # push analyzer to the end of the list
    file_list.sort(key=lambda s: s[1:])  # A3, B3, A4, an
    # file_list.sort(key=lambda s: s.startswith('an'))

    for filename in file_list:
        plts_name = filename[0:2]
        print('\n Reading geometry {} ...'.format(plts_name))
        r_new = geom.r_dict[plts_name]
        print('position ', r_new)
        # angles of plates, will be modified later
        plts_angles = copy.deepcopy(geom.angles_dict[plts_name])
        # beamline angles
        beamline_angles = copy.deepcopy(geom.angles_dict[plts_name])
        # read plates parameters from file
        edges_list = []
        with open(dirname + '/' + filename, 'r') as f:
            # read plates geometry, first remove comments '#', then convert to float
            plts_geom = [float(i) for i in f.readline().split('#')[0].split()]
            # read gamma angle (0 for Alpha and 90 for Beta plates)
            gamma = float(f.readline().split()[0])
            # xmin, xmax, ymin, ymax, zmin, zmax, delta
            domain = [float(i) for i in f.readline().split()[0:7]]
            if plts_name == 'an':
                an_params = [float(i) for i in f.readline().split()[0:8]]
                theta_an = an_params[4]  # analyzer entrance angle
                plts_angles[0] = plts_angles[0] - theta_an
            for line in f:
                # read plates Upper and Lowe plate coords, x,y,z
                edges_list.append([float(i) for i in line.split()[0:3]])
        edges_list = np.array(edges_list)

        if plts_name == 'an':
            # create new Analyzer object
            plts = Analyzer(plts_name, beamline)
            plts.add_slits(an_params)
        else:
            plts = Plates(plts_name, beamline)
        # add edges to plates object
        index = int(edges_list.shape[0] / 2)
        plts.set_edges(np.array([edges_list[0:index, :],
                                 edges_list[index:, :]]))
        # rotate plates edges
        plts.rotate(plts_angles, beamline_angles)
        # shift coords center and put into a dictionary
        plts.shift(r_new)
        # add plates to dictionary
        geom.plates_dict[plts_name] = plts

        # read Electric field arrays
        Ex = np.load(dirname + '/' + plts_name + '_Ex.npy')
        Ey = np.load(dirname + '/' + plts_name + '_Ey.npy')
        Ez = np.load(dirname + '/' + plts_name + '_Ez.npy')

        x = np.arange(domain[0], domain[1], domain[6])  # + r_new[0]
        y = np.arange(domain[2], domain[3], domain[6])  # + r_new[1]
        z = np.arange(domain[4], domain[5], domain[6])  # + r_new[2]

        # make interpolation for Ex, Ey, Ez
        Ex_interp = RegularGridInterpolator((x, y, z), Ex)
        Ey_interp = RegularGridInterpolator((x, y, z), Ey)
        Ez_interp = RegularGridInterpolator((x, y, z), Ez)
        E_read = [Ex_interp, Ey_interp, Ez_interp]

        E[plts_name] = E_read

    return


def read_B(Btor, Ipl, PF_dict, dirname='magfield', interp=True):
    '''
    read Magnetic field values and create Bx, By, Bz, rho interpolants
    '''
    print('\n Reading Magnetic field')
    B_dict = {}
    for filename in os.listdir(dirname):
        if filename.endswith('.dat'):
            with open(dirname + '/' + filename, 'r') as f:
                volume_corner1 = [float(i) for i in f.readline().split()[0:3]]
                volume_corner2 = [float(i) for i in f.readline().split()[0:3]]
                resolution = float(f.readline().split()[0])
            continue
        elif 'Tor' in filename:
            print('Reading toroidal magnetic field...')
            B_read = np.load(dirname + '/' + filename) * Btor
            name = 'Tor'

        elif 'Plasm' in filename:
            print('Reading plasma field...')
            B_read = np.load(dirname + '/' + filename)  # * Ipl
            name = 'Plasm'

        else:
            name = filename.replace('magfield', '').replace('.npy', '')
            print('Reading {} magnetic field...'.format(name))
            Icir = PF_dict[name]
            print('Current = ', Icir)
            B_read = np.load(dirname + '/' + filename) * Icir

        B_dict[name] = B_read

    # create grid of points
    grid = np.mgrid[volume_corner1[0]:volume_corner2[0]:resolution,
                    volume_corner1[1]:volume_corner2[1]:resolution,
                    volume_corner1[2]:volume_corner2[2]:resolution]

    B = np.zeros_like(B_read)
    for key in B_dict.keys():
        B += B_dict[key]

#    cutoff = 10.0
#    Babs = np.linalg.norm(B, axis=1)
#    B[Babs > cutoff] = [np.nan, np.nan, np.nan]

    # plot B stream
    hbplot.plot_B_stream(B, volume_corner1, volume_corner2, resolution, grid,
                         plot_sep=False, dens=3.0)

    x = np.arange(volume_corner1[0], volume_corner2[0], resolution)
    y = np.arange(volume_corner1[1], volume_corner2[1], resolution)
    z = np.arange(volume_corner1[2], volume_corner2[2], resolution)
    Bx = B[:, 0].reshape(grid.shape[1:])
    By = B[:, 1].reshape(grid.shape[1:])
    Bz = B[:, 2].reshape(grid.shape[1:])
    if interp:
        # make an interpolation of B
        Bx_interp = RegularGridInterpolator((x, y, z), Bx)
        By_interp = RegularGridInterpolator((x, y, z), By)
        Bz_interp = RegularGridInterpolator((x, y, z), Bz)
        print('Interpolants for magnetic field created')
        B_list = [Bx_interp, By_interp, Bz_interp]
    else:
        B_list = [Bx, By, Bz]

    return B_list


# %% poloidal field coils
def import_PFcoils(filename):
    '''
    import a dictionary with poloidal field coils parameters
    {'NAME': (x center, y center, width along x, width along y [m],
               current [MA-turn], N turns)}
    Andreev, VANT 2014, No.3
    '''
    d = {}  # defaultdict(list)
    with open(filename) as f:
        for line in f:
            if line[0] == '#':
                continue
            lineList = line.split(', ')
            key, val = lineList[0], tuple([float(i) for i in lineList[1:]])
            d[key] = val
    return d


def import_PFcur(filename, pf_coils):
    '''
    Creates dictionary with coils names and currents from TOKAMEQ file
    filename : Tokameqs filename
    pf_coils : coil dict (we only take keys)
    '''
    with open(filename, 'r') as f:
        data = f.readlines()  # read tokameq file
    PF_dict = {}  # Here we will store coils names and currents
    pf_names = list(pf_coils)  # get coils names
    n_coil = 0  # will be used for getting correct coil name
    for i in range(len(data)):
        if data[i].strip() == 'External currents:':
            n_line = i + 2  # skip 2 lines and read from the third
            break
    while float(data[n_line].strip().split()[3]) != 0:
        key = pf_names[n_coil]
        val = data[n_line].strip().split()[3]
        PF_dict[key] = float(val)
        n_line += 1
        n_coil += 1

    return PF_dict


# %%
def import_Bflux(filename):
    '''
    import magnetic flux from Tokameq file
    '''
    with open(filename, 'r') as f:
        data = f.readlines()

    # R coordinate corresponds to X, Z coordinate corresponds to Y
    NrNz = []
    for i in data[2].strip().split():
        if i.isdigit():
            NrNz.append(i)
    Nx = int(NrNz[0]) + 1
    Ny = int(NrNz[1]) + 1

    for i in range(len(data)):
        if ' '.join(data[i].strip().split()[:4]) == 'Flux at the boundary':
            bound_flux = float(data[i].strip().split()[-1])
        if data[i].strip() == 'Poloidal flux F(r,z)':
            index = i

    x_vals = [float(r) for r in data[index+1].strip().split()[1:]]
    x_vals = np.array(x_vals)

    Psi_data = [i.strip().split() for i in data[index+2:index+2+Ny]]
    Psi_vals = []
    y_vals = []
    for line in Psi_data:
        y_vals.append(float(line[0]))
        Psi_vals.append([float(j) for j in line[1:]])

    y_vals = np.array(y_vals)
    Psi_vals = np.array(Psi_vals)
    return Psi_vals, x_vals, y_vals, bound_flux


# %%
def save_traj_list(traj_list, Btor, Ipl, r_aim, dirname='output'):
    '''
    save list of Traj objects to *.pkl file
    '''

    if len(traj_list) == 0:
        print('traj_list empty! nothing to save')
        return

    Ebeam_list = []
    UA2_list = []

    for traj in traj_list:
        Ebeam_list.append(traj.Ebeam)
        UA2_list.append(traj.U['A2'])

    dirname = dirname + '/' + 'B{}_I{}'.format(int(Btor), int(Ipl))

    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o700)
            print("Directory ", dirname, " created ")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    fname = dirname + '/' + 'E{}-{}'.format(int(min(Ebeam_list)),
                                            int(max(Ebeam_list))) + \
        '_UA2{}-{}'.format(int(min(UA2_list)), int(max(UA2_list))) + \
        '_alpha{:.1f}_beta{:.1f}'.format(traj.alpha, traj.beta) +\
        '_x{}y{}z{}.pkl'.format(int(r_aim[0]*100), int(r_aim[1]*100),
                                int(r_aim[2]*100))

    with open(fname, 'wb') as f:
        pc.dump(traj_list, f, -1)

    print('\nSAVED LIST: \n' + fname)


# %%
def read_traj_list(fname, dirname='output'):
    '''
    import list of Traj objects from *.pkl file
    '''
    with open(dirname + '/' + fname, 'rb') as f:
        traj_list = pc.load(f)
    return traj_list


# %%
def save_traj2dat(traj_list, save_fan=False, dirname='output/',
                  fmt='%.2f', delimiter=' '):
    '''
    save list of trajectories to *.dat files for CATIA plot
    '''
    for tr in traj_list:
        # save primary
        fname = dirname + 'E{:.0f}_U{:.0f}_prim.dat'.format(tr.Ebeam, tr.U['A2'])
        np.savetxt(fname, tr.RV_prim[:, 0:3]*1000,
                   fmt=fmt, delimiter=delimiter)  # [mm]
        # save secondary
        fname = dirname + 'E{:.0f}_U{:.0f}_sec.dat'.format(tr.Ebeam, tr.U['A2'])
        np.savetxt(fname, tr.RV_sec[:, 0:3]*1000,
                   fmt=fmt, delimiter=delimiter)


# %%
def save_png(fig, name, save_dir='output'):
    '''
    saves picture as name.png
    fig : array of figures to save
    name : array of picture names
    save_dir : directory used to store results
    '''

    # check wether directory exist and if not - create one
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print('LOG: {} directory created'.format(save_dir))
    print('LOG: Saving pictures to {}'.format(save_dir+'/'))
    for fig, name in zip(fig, name):
        # save fig with tight layout
        fig_savename = str(name + '.png')
        fig.savefig(save_dir + '/' + fig_savename, bbox_inches='tight')
        print('LOG: Figure ' + fig_savename + ' saved')
