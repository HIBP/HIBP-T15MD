import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import hibplib as hb
import hibpplotlib as hbplot

# %% import trajectory lists
if __name__ == '__main__':
    traj_list = []

    names = ['E100-120_UA20-21_alpha30_beta-10_x260y-25z0.pkl',
             'E140-160_UA2-3-24_alpha30_beta-10_x260y-25z0.pkl',
             'E180-200_UA2-3-24_alpha30_beta-10_x260y-25z0.pkl',
             'E220-240_UA2-3-30_alpha30_beta-10_x260y-25z0.pkl',
             'E260-280_UA29-30_alpha30_beta-10_x260y-25z0.pkl',
             'E300-300_UA227-30_alpha30_beta-10_x260y-25z0.pkl']

    for name in names:
        traj_list += hb.read_traj_list(name, dirname='output/B1_I1')
