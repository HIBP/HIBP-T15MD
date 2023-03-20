"""
Created on Thu Feb  9 14:06:43 2023


??? look on following !!!
interpn
    a convenience function which wraps RegularGridInterpolator


scipy.ndimage.map_coordinates
    interpolation on grids with equal spacing (suitable for e.g., N-D image resampling)


@author: reonid

_numba_njit = numba.njit

"""

import numpy as np
import copy
import time
import sys
import os

#import builtins
#print = builtins.print

import matplotlib.pyplot as plt

from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates

import hibplib as hb
import define_geometry as defgeom
#import hibpplotlib as hbplot
#import define_geometry as defgeom

#import hibplib.physconst as physconst

MAX_TRAJ_LEN = 5000

'''
class EqualSpacingGridInterpolator: 
#    def __init__(self, points, values, method="linear", bounds_error=True, fill_value=np.nan): 
    def __init__(self, points, values, method="linear", bounds_error=True, fill_value=np.nan): 
        self.points = points
        self.values = values
        # self.bounds_error = bounds_error
        # self.method = method
        self.fill_value = fill_value 
        
        self.x0 = points[0, 0, 0][0]
        self.y0 = points[0, 0, 0][1]
        self.y0 = points[0, 0, 0][2]
        
        self.x1 = points[-1, -1, -1][0]
        self.y1 = points[-1, -1, -1][1]
        self.y1 = points[-1, -1, -1][2]
    
    #def __call__(self, self, xi, method=None): 
    def __call__(self, self, xi): 
        x, y, z = xi
        i = x
        j = y
        k = z
        
        return map_coordinates(self.values, [[i],[j],[k]], order=0, cval=self.fill_value)
'''

def read_grid_from_dat(filename): 
    '''
    geometry.dat
    '''
    with open(filename, 'r') as f: 
        volume_corner1 = [float(s) for s in f.readline().split()[0:3]]
        volume_corner2 = [float(s) for s in f.readline().split()[0:3]]
        resolution = float(f.readline().split()[0])    

    grid = np.mgrid[volume_corner1[0]:volume_corner2[0]:resolution,
                    volume_corner1[1]:volume_corner2[1]:resolution,
                    volume_corner1[2]:volume_corner2[2]:resolution]

    xx = np.arange(volume_corner1[0], volume_corner2[0], resolution)
    yy = np.arange(volume_corner1[1], volume_corner2[1], resolution)
    zz = np.arange(volume_corner1[2], volume_corner2[2], resolution)

    return grid, xx, yy, zz

"""
def read_mag_npy_list(path): 
    '''
    returns list of (ident, filename)
    '''
    result = []
    for filename in os.listdir(path): 
        if 'old' in filename:
            continue # ??? 
        elif ('magfield' in filename) and ('.npy' in filename):
            ident = filename.replace('magfield', '').replace('.npy', '')
            ident = ident.split('_')[0]
            result.append((ident, filename))
    return result
"""


def read_E(): 
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
    
    return E    


class BField: 
    def __init__(self, Btor, Ipl, PF_dict, dirname='magfield'):
        '''
        read Magnetic field values and create Bx, By, Bz, rho interpolants
        '''
        print('\n Reading Magnetic field')
        B_dict = {}
        #grid, xx, yy, zz = read_grid_from_dat(dirname + '/geometry.dat')
    
        for filename in os.listdir(dirname):
            if 'old' in filename:
                continue 
            elif filename.endswith('.dat'): 
                grid, xx, yy, zz = read_grid_from_dat(dirname + '/' + filename)
                continue 
            elif 'Tor' in filename:
                print('Reading toroidal magnetic field...')
                B_read = np.load(dirname + '/' + filename) * Btor
                name = 'Tor'
    
            elif 'Plasm_{}MA'.format(int(Ipl)) in filename:
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
    
        B = np.zeros_like(B_read)
        for key in B_dict.keys():
            B += B_dict[key]
    
    #    cutoff = 10.0
    #    Babs = np.linalg.norm(B, axis=1)
    #    B[Babs > cutoff] = [np.nan, np.nan, np.nan]
    
        _Bx = B[:, 0].reshape(grid.shape[1:])
        _By = B[:, 1].reshape(grid.shape[1:])
        _Bz = B[:, 2].reshape(grid.shape[1:])
    
        # make an interpolation of B
        Bx_interp = RegularGridInterpolator((xx, yy, zz), _Bx, bounds_error=False)
        By_interp = RegularGridInterpolator((xx, yy, zz), _By, bounds_error=False)
        Bz_interp = RegularGridInterpolator((xx, yy, zz), _Bz, bounds_error=False)
        print('Interpolants for magnetic field created')
        #B_list = [Bx_interp, By_interp, Bz_interp]
        

    
        self.Bx = Bx_interp
        self.By = By_interp
        self.Bz = Bz_interp
        

    def __call__(self, point): 
        return np.c_[self.Bx(point), self.By(point), self.Bz(point)] 
        #return np.r_[self.Bx(point), self.By(point), self.Bz(point)] 

class Efield: 
    def __init__(self, Ein, U, geom): 
        self.Ein = Ein
        self.U = U
        self.geom = geom

    #def return_E(r, Ein, U, geom):
    def __call__(self, r): 
        '''
        take dot and try to interpolate electiric field
        Ein : dict of interpolants for Ex, Ey, Ez
        U : dict with plates voltage values
        '''
        Ein = self.Ein
        U = self.U
        geom = self.geom
                
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
            r_new = hb.rotate3(r_new, angles, beamline_angles, inverse=True)
            # interpolate Electric field
            Etemp = np.zeros(3)
            try:
                Etemp[0] = Ein[key][0](r_new) * U[key]
                Etemp[1] = Ein[key][1](r_new) * U[key]
                Etemp[2] = Ein[key][2](r_new) * U[key]
                # rotate Etemp
                Etemp = hb.rotate3(Etemp, angles, beamline_angles, inverse=False)
                # add the result to total E field
                Etotal += Etemp
            except (ValueError, IndexError):
                continue
        return Etotal

def f(k, E, V, B):
    return k*(E + np.cross(V, B))


def g(V):
    return V


def _runge_kutt(k, RV, dt, E, B):
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
    
    if np.any(np.isnan(B)): 
        print('NaN!!  B = ', B)
        print('   RV = ', RV)
    
    if np.any(np.isnan(E)): 
        print('NaN!!  E = ', E)
        print('   RV = ', RV)
    
    #r = RV[0, :3]
    #V = RV[0, 3:]

    r = RV[:3]
    V = RV[3:]

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
    #RV = np.r_[r, V]
    return RV[0, :]

class Trajectory():
    def __init__(self, q, m, Ebeam, r0, alpha, beta, dt=1e-7):  # U, 
        self.q = q
        self.m = m
        self.Ebeam = Ebeam
        self.r0 = r0
        Vabs = np.sqrt(2.0 * Ebeam * 1.602176634E-16 / m)
        self.v0 = hb.calc_vector(-Vabs, alpha, beta)
        self.rv0 = np.hstack((self.r0, self.v0))
        self.dt = dt
        
        self.rrvv = None # np.empry((0, 6))
    
    def pass_(self, E, B, stop): 
        q_m = self.q / self.m
        rv  = self.rv0 #np.r_[self.r0, self.v0] 
        dt = self.dt
        t = 0.0 # ???

        _rrvv = np.empty( (MAX_TRAJ_LEN, 6) )
        _curr = 0

        _rrvv[_curr] = rv; _curr += 1 
 
        while True: # t < t_max
            r = rv[0:3]
            E_loc = E(r)
            B_loc = B(r)
            if np.isnan(B_loc[0, 0]): break
        
            #rv_new = hb.runge_kutt(q_m, rv, dt, E_loc, B_loc)
            rv_new = _runge_kutt(q_m, rv, dt, E_loc, B_loc)
            _rrvv[_curr] = rv_new; _curr += 1 
            
            if stop(rv, rv_new): break
            
            rv = rv_new
            t = t + dt
        
        self.rrvv = _rrvv[0:_curr]

    def plot(self, *args, **kwargs): 
        xx = self.rrvv[:, 0] 
        yy = self.rrvv[:, 1] 
        plt.plot(xx, yy, *args, **kwargs)

#B = hb.read_B(Btor, Ipl, PF_dict, dirname=dirname, plot=plot_B)

Btor = 1.5  # [T]
Ipl = 1.0   # Plasma current [MA]


# %% Define Geometry
geomT15 = defgeom.define_geometry(analyzer=1)

# angles of aim plane normal [deg]
alpha_aim = 0.
beta_aim = 0.
stop_plane_n = hb.calc_vector(1.0, alpha_aim, beta_aim)

# %% Load Magnetic Field
pf_coils = hb.import_PFcoils('PFCoils.dat')
PF_dict = hb.import_PFcur('{}MA_sn.txt'.format(int(abs(Ipl))), pf_coils)

B = BField(Btor, Ipl, PF_dict)

pt = np.array([2.8, 0.0, 0.0])
print(B(pt))

# UA2 voltages
UA2 = 10.  #-50., 50., 2. #12., 12., 2. #-50., 50., 2.  #0., 34., 2.  # -3, 33., 3.  # -3., 30., 3.
UB2 = 0.0  # 10.  # [kV], [kV/m]
UB3 = 0.3  # [kV], [kV/m]
UA3 = 0.4  # [kV], [kV/m]
UA4 = 0.1  # [kV], [kV/m]
Uan = 50.  # Ebeam/(2*G)
U_dict = {'A2': UA2, 'B2': UB2, 'A3': UA3, 'B3': UB3, 'A4': UA4, 'an': Uan}

_E = read_E()
E = Efield(_E, U_dict, geomT15)

PI = 3.1415926535897932384626433832795

SI_AEM     = 1.6604e-27       #     {kg}
SI_e       = 1.6021e-19       #     {C}
SI_Me      = 9.1091e-31       #     {kg}      // mass of electron
SI_Mp      = 1.6725e-27       #     {kg}      // mass of proton
SI_c       = 2.9979e8         #     {m/sec}   // velocity of light
SI_1eV     = SI_e             #     {J}
SI_1keV    = SI_1eV*1000.0    #     {J}


dt = 0.2e-7 
tr = Trajectory(SI_e, SI_AEM*204.0, 180.0, geomT15.r_dict['r0'], geomT15.angles_dict['r0'][0], 
             geomT15.angles_dict['r0'][1], dt)

class StopPrim: 
    def __init__(self, geom, invisible_wall_x = 5.5): 
        self.geom = geom
        self.invisible_wall_x = invisible_wall_x
        #invisible_wall_x = self.geom.r_dict[target][0]+0.2
        
    def __call__(self, rv_old, rv_new): 

        if rv_new[0] > self.invisible_wall_x and rv_new[1] < 1.2:
            return True

        if self.geom.check_chamb_intersect('prim', rv_old[0:3], rv_new[0:3]):
            return True

        plts_flag, plts_name = self.geom.check_plates_intersect(rv_old[0:3], rv_new[0:3])
        if plts_flag:
            return True

        if self.geom.check_fw_intersect(rv_old[0:3], rv_new[0:3]):
            return True  # stop primary trajectory calculation        
        
        return False

class StopSec: 
    def __init__(self, geom, r_aim, invisible_wall_x = 5.5): 
        self.geom = geom
        self.r_aim = r_aim
        self.invisible_wall_x = invisible_wall_x
        #invisible_wall_x = self.geom.r_dict[target][0]+0.2
        
        self.rrvv = np.empty( (0, 6) )

        
    def __call__(self, rv_old, rv_new):
        r_aim = self.r_aim
        eps_xy = 1e-3
        eps_z = 1e-3

        if rv_new[0] > self.invisible_wall_x: 
            return True
            
        if self.geom.check_chamb_intersect('sec', rv_old[0:3], rv_new[0:3]):
            #self.IntersectGeometrySec['chamb'] = True
            return True

        plts_flag, plts_name = self.geom.check_plates_intersect(rv_old[0:3], rv_new[0:3])
        if plts_flag:
            #self.IntersectGeometrySec[plts_name] = True
            return True

        # find last point of the secondary trajectory
        if (rv_new[0] > 2.5) and (rv_new[1] < 1.5):
            # intersection with the stop plane:
            r_intersect = hb.line_plane_intersect(stop_plane_n, r_aim, rv_new[:3] - rv_old[:3], rv_new[:3])
            # check if r_intersect is between RV_old and RV_new:
            if hb.is_between(rv_old[:3], rv_new[:3], r_intersect):
                rv_new[:3] = r_intersect
                self.rrvv = np.vstack((self.rrvv, rv_new))  # ???
                # check XY plane:
                if (np.linalg.norm(rv_new[:2] - r_aim[:2]) <= eps_xy):
                    # print('aim XY!')
                    # self.IsAimXY = True
                    return True
                # check XZ plane:
                if (np.linalg.norm(rv_new[0, [0, 2]] - r_aim[[0, 2]]) <= eps_z):
                    # print('aim Z!')
                    # self.IsAimZ = True
                    return True
            
            return False
    
tr.pass_(E, B, stop=StopPrim(geomT15))
tr.plot()


'''

    B_list = read_mag_npy_list(dirname)
    B = None
    for ident, filename in B_list: 
        if ident == "Tor": 
            coeff = Btor
        else:     
            coeff = PF_dict[ident]
        B_data = np.load(dirname + '/' + filename) * coeff
        if B is None: 
            B = np.zeros_like(B_data)
        else:
            B += B_data



class HibpConf: 
    def __init__(self, geom): 
        self.geom = geom
        self.plates_U = dict()
        self.plates = dict()
    
    def E(self): 
        return Efield(self.)
'''
