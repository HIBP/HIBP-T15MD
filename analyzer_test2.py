import numpy as np
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %%
''' pass trajectories to detector inside the analyzer
'''
Ebeam = 240.
UA2 = 0.

n_slits = geomT15.slits_edges.shape[0]

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

    tr = hb.pass_to_slits(tr, dt, E, B, geomT15,
                          target='slit', timestep_divider=10,
                          no_intersect=True, no_out_of_bounds=True)
    hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl, plot_fan=True)


    # tr = hb.pass_to_slits(tr, dt, E, B, geomT15,
    #                       target='det', timestep_divider=15,
    #                       no_intersect=True, no_out_of_bounds=True)
    # hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl, plot_fan=True)

    print('\n Passing to Analyzer')
    for i_slit in range(n_slits):
        rv_list = []
        for fan_tr in tr.RV_sec_toslits[i_slit]:
            RV0 = np.array([fan_tr[0]])
            # pass traj to detector
            tr.pass_sec(RV0, geomT15.r_dict['det'], E, B, geomT15, tmax=9e-5,
                        eps_xy=1, eps_z=1)
            if not (tr.IntersectGeometrySec):  # or tr.B_out_of_bounds):
                rv_list.append(tr.RV_sec)
        # update RV list
        tr.RV_sec_toslits[i_slit] = rv_list

    break

# %% plot trajectories

# hbplot.plot_traj(traj_list_copy, geomT15, 240., UA2, Btor, Ipl,
#                   full_primary=False, plot_analyzer=True)
hbplot.plot_traj_toslits(tr, geomT15, Btor, Ipl, plot_fan=True)
