import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %%
''' pass trajectories to detector inside the analyzer
'''
Ebeam = 240.
UA2 = 9.

# %%
traj_list_copy = copy.deepcopy(traj_list_a3b3)
# traj_list_copy = copy.deepcopy(traj_list_passed)

# %%
print('\n*** Passing traj to Analyzer'.format(n_slits))
for tr in traj_list_copy:
    if tr.Ebeam == Ebeam and tr.U[0] == UA2:
        print('\nEb = {}, UA2 = {}'.format(tr.Ebeam, tr.U[0]))
    else:
        continue
    RV0 = np.array([tr.RV_sec[0]])
    # tr.dt2 = 7e-8/4
    tr.pass_sec(RV0, geomT15.r_dict['det'], E, B, geomT15,
                stop_plane_n=geomT15.det_plane_n,
                tmax=9e-5, eps_xy=1e-3, eps_z=1)
    # tr = hb.pass_to_slits(tr, dt, E, B, geomT15,
    #                       target='det', timestep_divider=15)

    break

# %% plot trajectories

hbplot.plot_traj(traj_list_copy, geomT15, 240., UA2, Btor, Ipl,
                 full_primary=False, plot_analyzer=True,
                 subplots_vertical=True, scale=3.5)
hbplot.plot_scan(traj_list_copy, geomT15, 240., Btor, Ipl,
                 full_primary=False, plot_analyzer=True,
                 plot_det_line=False, subplots_vertical=True, scale=3.5)
# hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl, plot_fan=True)
