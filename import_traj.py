import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import hibplib as hb
import hibpplotlib as hbplot
import copy

# %% import trajectory lists
if __name__ == '__main__':
    traj_list = []

    # grid with alpha=34 y_aim=-10
    # names = ['E100-140_UA26-33_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E160-200_UA26-33_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E220-260_UA23-33_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E280-320_UA212-33_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E340-340_UA230-33_alpha34.0_beta-10.0_x260y-10z1.pkl']
    # grid with alpha=34 y_aim=05
    # names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x260y5z1.pkl',
    #          'E160-200_UA26-24_alpha34.0_beta-10.0_x260y5z1.pkl',
    #          'E220-260_UA212-33_alpha34.0_beta-10.0_x260y5z1.pkl',
    #          'E280-300_UA230-33_alpha34.0_beta-10.0_x260y5z1.pkl']
    # grid with alpha=34 y_aim=10
    # names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x260y10z1.pkl',
    #          'E160-200_UA26-24_alpha34.0_beta-10.0_x260y10z1.pkl',
    #          'E220-260_UA218-33_alpha34.0_beta-10.0_x260y10z1.pkl']
    # grid with alpha=34 y_aim=0
    # names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x260y0z1.pkl',
    #          'E160-200_UA26-24_alpha34.0_beta-10.0_x260y0z1.pkl',
    #          'E220-260_UA26-33_alpha34.0_beta-10.0_x260y0z1.pkl',
    #          'E280-300_UA224-33_alpha34.0_beta-10.0_x260y0z1.pkl']
    # grid with alpha=34 y_aim=-5
    # names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x260y-5z1.pkl',
    #          'E160-200_UA26-24_alpha34.0_beta-10.0_x260y-5z1.pkl',
    #          'E220-260_UA23-33_alpha34.0_beta-10.0_x260y-5z1.pkl',
    #          'E280-320_UA218-33_alpha34.0_beta-10.0_x260y-5z1.pkl']
    # grid with alpha=34 y_aim=-10
    # names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E160-200_UA26-24_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E220-260_UA23-33_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E280-320_UA212-33_alpha34.0_beta-10.0_x260y-10z1.pkl',
    #          'E340-340_UA230-33_alpha34.0_beta-10.0_x260y-10z1.pkl']
    # grid with alpha=34 y_aim=-15
    # names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x260y-15z1.pkl',
    #          'E160-200_UA23-24_alpha34.0_beta-10.0_x260y-15z1.pkl',
    #          'E220-260_UA23-30_alpha34.0_beta-10.0_x260y-15z1.pkl',
    #          'E280-320_UA23-33_alpha34.0_beta-10.0_x260y-15z1.pkl',
    #          'E340-380_UA224-33_alpha34.0_beta-10.0_x260y-15z1.pkl']
    # grid with alpha=34 y_aim=-20
    # names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x260y-20z1.pkl',
    #          'E160-200_UA23-24_alpha34.0_beta-10.0_x260y-20z1.pkl',
    #          'E220-260_UA23-33_alpha34.0_beta-10.0_x260y-20z1.pkl',
    #          'E280-320_UA23-33_alpha34.0_beta-10.0_x260y-20z1.pkl',
    #          'E340-380_UA215-33_alpha34.0_beta-10.0_x260y-20z1.pkl']
    
    # grid with alpha=34 y_aim=-10, test
    names = ['E100-140_UA26-18_alpha34.0_beta-10.0_x250y-10z5.pkl',
             'E200-200_UA24-26_alpha34.0_beta-10.0_x250y-10z3.pkl']

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
