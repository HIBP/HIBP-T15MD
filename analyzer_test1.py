import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %%
''' pass trajectories to detector inside the analyzer
'''
Ebeam = 240.
UA2 = 5.0

n_slits, slit_dist, slit_w = geomT15.an_params[:3]
# add slits to Geometry
geomT15.add_slits(n_slits=n_slits, slit_dist=slit_dist, slit_w=slit_w,
                  slit_l=0.1)
r_slits = geomT15.slits_edges
rs = geomT15.r_dict['slit']
# calculate normal to slit plane
slit_plane_n = geomT15.slit_plane_n

# define detector
geomT15.add_detector(n_det=n_slits, det_dist=slit_dist, det_w=slit_dist,
                     det_l=0.1)

# %%
traj_list_copy = copy.deepcopy(traj_list_a3b3)
# traj_list_copy = copy.deepcopy(traj_list_passed)

# %%
print('\n*** Passing fan to {} slits'.format(n_slits))
for tr in traj_list_copy:
    if tr.Ebeam == Ebeam and tr.U[0] == UA2:
        print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U[0]))
    else:
        continue

    tr = hb.pass_to_slits(tr, dt, E, B, geomT15, timestep_divider=15)
    break

# %% plot trajectories
hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl, plot_fan=True)
