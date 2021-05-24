import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %%
''' pass trajectories to different slits
'''
Ebeam = 240.
UA2 = 9.0

n_slits = 7
# add slits to Geometry
# geomT15.add_slits(n_slits=n_slits, slit_dist=0.01, slit_w=5e-3,
#                   slit_l=0.1)  # -20.)
r_slits = geomT15.slits_edges
rs = geomT15.r_dict['slit']
# calculate normal to slit plane
slit_plane_n = geomT15.slit_plane_n

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

    tr = hb.pass_to_slits(tr, dt, E, B, geomT15, timestep_divider=5)
    break

# %% plot trajectories
hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl, plot_fan=True)
