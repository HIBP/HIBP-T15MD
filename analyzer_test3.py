import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import hibplib as hb
import hibpplotlib as hbplot
import copy
import time


def Bzero(r):
    return 0.


# %% ANALYZER test

# timestep [sec]
dt = 0.025e-7  # 0.4e-7  # 0.7e-7

q = 1.60217662e-19  # electron charge [Co]
m_ion = 204.3833 * 1.6605e-27  # Tl ion mass [kg]

# initial beam energy range
Ebeam = 240.
UA2 = 0.

# %% PRIMARY beamline geometry
geom_test = hb.Geometry()

# plasma parameters
geom_test.R = 1.5  # tokamak major radius [m]
geom_test.r_plasma = 0.7  # plasma minor radius [m]
geom_test.elon = 1.8  # plasma elongation

# %% AIM position (BEFORE the Secondary beamline) [m]
xaim = 0.  # 2.6  # 2.5
yaim = 0.  # -0.25
zaim = 0.
r_aim = np.array([xaim, yaim, zaim])
geom_test.r_dict['aim'] = r_aim

# %% SECONDARY beamline geometry
# alpha and beta angles of the SECONDARY beamline [deg]
alpha_sec = 30.  # 20.
beta_sec = 0.  # 20.
gamma_sec = 0.  # -20.
geom_test.sec_angles = np.array([alpha_sec, beta_sec, gamma_sec])

# distance from r_aim the entrance slit of the analyzer
dist_s = 0.05

# Coordinates of the CENTRAL slit
geom_test.add_coords('slit', 'aim', dist_s, geom_test.sec_angles)
# Coordinates of the ANALYZER
geom_test.add_coords('an', 'aim', dist_s, geom_test.sec_angles)

# %% print info
print('r_aim = ', r_aim)
print('r_slit = ', np.round(geom_test.r_dict['slit'], 3))

# %% Load Electric Field
''' Electric field part '''

# load E for secondary beamline
try:
    E_sec, edges_sec = hb.read_E('sec', geom_test)
    geom_test.plates_edges.update(edges_sec)
except FileNotFoundError:
    print('Secondary Beamline NOT FOUND')
    E_sec = []

E = E_sec

# %% Analyzer parameters
if geom_test.an_params.shape[0] > 0:
    # Analyzer G
    G = geom_test.an_params[3]

    n_slits, slit_dist, slit_w = geom_test.an_params[:3]

    # add slits to Geometry
    geom_test.add_slits(n_slits=n_slits, slit_dist=slit_dist, slit_w=slit_w,
                      slit_l=0.1)

    # define detector
    geom_test.add_detector(n_det=n_slits, det_dist=slit_dist,
                         det_w=slit_dist, det_l=0.1)
    print('\nAnalyzer with {} slits added to Geometry'.format(n_slits))
    print('G = {}\n'.format(G))
else:
    print('\nNO Analyzer')

# %% Load Magnetic Field
''' Magnetic field part '''
B = [Bzero, Bzero, Bzero]

# %% Optimize Primary Beamline
print('\n Primary beamline optimization')
t1 = time.time()

print('\n\nE = {} keV; UA2 = {} kV\n'.format(Ebeam, UA2))
# list of starting voltages
U_list = [Ebeam/(2*G)]

traj_list_test = []
# create new trajectory object
for i_slit in range(int(n_slits)):
    r0 = copy.deepcopy(r_aim)
    r0[1] += (n_slits//2 - i_slit)*slit_dist  #  / np.cos(30. * np.pi/180.)
    r0 = hb.rotate(r0, axis=(0, 0, 1), deg=alpha_sec)
    tr = hb.Traj(2*q, m_ion, Ebeam, r0, 180 + alpha_sec, beta_sec,
                 U_list, dt)
    tr.pass_prim(E, B, geom_test, tmax=2e-6)
    traj_list_test.append(tr)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)

hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')
hbplot.set_axes_param(ax2, 'X (m)', 'Z (m)')
hbplot.set_axes_param(ax3, 'Z (m)', 'Y (m)')
ax1.set_title('E={} keV, Uan={} kV'
              .format(tr.Ebeam, tr.U[0]))

# draw starting point
ax1.plot(r_aim[0], r_aim[1], '*')
ax2.plot(r_aim[0], r_aim[2], '*')
ax3.plot(r_aim[2], r_aim[1], '*')

# draw slits
geom_test.plot_analyzer(ax1, axes='XY')
geom_test.plot_analyzer(ax2, axes='XZ')
geom_test.plot_analyzer(ax3, axes='ZY')

n_slits = geom_test.slits_edges.shape[0]
# set color cycler
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
colors = colors[:n_slits]
colors = cycle(colors)

for tr in traj_list_test:
    c = next(colors)
    # plot primary trajectory
    tr.plot_prim(ax1, axes='XY', color=c, full_primary=True)
    tr.plot_prim(ax2, axes='XZ', color=c, full_primary=True)
    tr.plot_prim(ax3, axes='ZY', color=c, full_primary=True)
