import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %% import trajectory lists
if __name__ == '__main__':
    traj_list = []

    # names = ['E100-120_UA20-21_alpha30_beta-10_x260y-25z0.pkl',
    #          'E140-160_UA2-3-24_alpha30_beta-10_x260y-25z0.pkl',
    #          'E180-200_UA2-3-24_alpha30_beta-10_x260y-25z0.pkl',
    #          'E220-240_UA2-3-30_alpha30_beta-10_x260y-25z0.pkl',
    #          'E260-280_UA29-30_alpha30_beta-10_x260y-25z0.pkl',
    #          'E300-300_UA227-30_alpha30_beta-10_x260y-25z0.pkl']

    names = ['E100-120_UA20-18_alpha30_beta-10_x260y-25z0.pkl',
             'E140-160_UA2-3-24_alpha30_beta-10_x260y-25z0.pkl',
             'E180-200_UA2-3-30_alpha30_beta-10_x260y-25z0.pkl',
             'E220-240_UA2-3-30_alpha30_beta-10_x260y-25z0.pkl',
             'E260-280_UA29-30_alpha30_beta-10_x260y-25z0.pkl',
             'E300-320_UA224-30_alpha30_beta-10_x260y-25z0.pkl']

    # names = ['E100-120_UA2-3-18_alpha20_beta-10_x260y-25z0.pkl',
    #           'E140-160_UA2-6-18_alpha20_beta-10_x260y-25z0.pkl',
    #           'E180-200_UA2-9-12_alpha20_beta-10_x260y-25z0.pkl',
    #           'E220-240_UA2-9-15_alpha20_beta-10_x260y-25z0.pkl',
    #           'E260-280_UA23-30_alpha20_beta-10_x260y-25z0.pkl',
    #           'E300-320_UA215-30_alpha20_beta-10_x260y-25z0.pkl']

    for name in names:
        traj_list += hb.read_traj_list(name, dirname='output/B1_I1')

# %%
    traj_list_passed = copy.deepcopy(traj_list)

# %% Save traj list
    Btor = 1.0
    Ipl = 1.0
    r_aim = geomT15.r_dict['aim']
    hb.save_traj_list(traj_list_passed, Btor, Ipl, r_aim)

# %% Additonal plots

    hbplot.plot_grid(traj_list_passed, geomT15, Btor, Ipl, marker_A2='')
    # hbplot.plot_fan(traj_list_passed, geomT15, 240., UA2, Btor, Ipl,
    #                 plot_analyzer=False, plot_traj=True, plot_all=True)
