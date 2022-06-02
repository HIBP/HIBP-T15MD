import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
import hibplib as hb
import hibpplotlib as hbplot
import matplotlib.colors as colors


# %%
# Calculate HIBP beam attenuation in T-15MD tokamak
def egg_fun(x, const=0.8):
    '''function transforms ellipse to egg-like curve
    '''
    return (1 + const*x)


def get_rho(x, y, z, r_pl=0.65, elon=1.7, R=1.5):
    '''
    function returns normalized radius
    R - major radius
    elon - elongation along y
    r_pl - plasma radius along x
    '''
    x1 = np.sqrt(x**2 + z**2)
    return np.sqrt((x1-R)**2 + egg_fun(x1-R)*(y/elon)**2) / r_pl


def ne(rho, ne0):
    '''
    plasma electron density profile
    rho - normalized radius
    ne0 - central density
    '''
    return ne0*(1.0 - rho**2)


def Te(rho, Te0):
    '''
    plasma Te profile
    rho - normalized radius
    Te0 - central temperature
    '''
    return Te0/(1 + (rho/0.5)**2)**(4/3)
#    return Te0*rho/rho


def integrate_traj(tr, ne0, Te0, sigmaEff12, sigmaEff23):
    '''
    ne0, Te0 - central values
    sigmaV - interpolant over Te
    returns ne(rho)*(sigma_V(rho)/v0)*lam*exp(-integral_prim-integral_sec)
    '''
    # initial particle velocity [m/s]
    v0 = math.sqrt(2*tr.Ebeam*1.6e-16 / tr.m)

    # first of all add the first point of secondary traj to the primary traj
    # find the distances
    distances = np.array([np.linalg.norm(tr.RV_prim[i, :3] - tr.RV_sec[0, :3])
                         for i in range(tr.RV_prim.shape[0])])
    sorted_indices = np.argsort(distances)
    # find position where to insert the new point
    index_to_insert = max(sorted_indices[0:2])
    tr.RV_prim = np.insert(tr.RV_prim, index_to_insert,
                           tr.RV_sec[0, :], axis=0)

    # integrals over primary and secondary trajectory
    I1, I2, L1, L2 = 0., 0., 0., 0.
    # integration loop
    # integrating primary trajectory
    for i in range(1, index_to_insert+1):
        x1, y1, z1 = tr.RV_prim[i-1, 0], tr.RV_prim[i-1, 1], tr.RV_prim[i-1, 2]
        x2, y2, z2 = tr.RV_prim[i, 0], tr.RV_prim[i, 1], tr.RV_prim[i, 2]

        rho1 = get_rho(x1, y1, z1, r_pl=r_plasma, elon=elon, R=R)
        rho2 = get_rho(x2, y2, z2, r_pl=r_plasma, elon=elon, R=R)

        if (rho1 <= 1) & (rho2 <= 1):
            dl = np.linalg.norm([x2-x1, y2-y1, z2-z1])
            r_loc = (rho1 + rho2) / 2
            ne_loc = 1e19 * ne(r_loc, ne0)
            Te_loc = Te(r_loc, Te0)
            I1 += dl * sigmaEff12(Te_loc) * ne_loc / v0
            L1 += dl

    # integrating secondary trajectory
    for i in range(1, tr.RV_sec.shape[0]):
        x1, y1, z1 = tr.RV_sec[i-1, 0], tr.RV_sec[i-1, 1], tr.RV_sec[i-1, 2]
        x2, y2, z2 = tr.RV_sec[i, 0], tr.RV_sec[i, 1], tr.RV_sec[i, 2]

        rho1 = get_rho(x1, y1, z1, r_pl=r_plasma, elon=elon, R=R)
        rho2 = get_rho(x2, y2, z2, r_pl=r_plasma, elon=elon, R=R)

        if (rho1 <= 1) & (rho2 <= 1):
            dl = np.linalg.norm([x2-x1, y2-y1, z2-z1])
            r_loc = (rho1 + rho2) / 2
            ne_loc = 1e19 * ne(r_loc, ne0)
            Te_loc = Te(r_loc, Te0)
            I2 += dl * sigmaEff23(Te_loc) * ne_loc / v0
            L2 += dl

    r_loc = get_rho(tr.RV_sec[0, 0], tr.RV_sec[0, 1], tr.RV_sec[0, 2])
    if r_loc <= 0.99:
        Te_loc = Te(r_loc, Te0)
        ne_loc = 1e19 * ne(r_loc, ne0)
        sigmaEff_loc = sigmaEff12(Te_loc) / v0
    else:
        Te_loc = 0.1  # 0.
        ne_loc = 1e19 * ne0 * 1e-2  # 0.
        sigmaEff_loc = sigmaEff12(Te_loc) / v0  # 0.

    # calculate total value with integrals
    lam = 0.005  # [m]
    Itot = 2 * ne_loc * sigmaEff_loc * lam * math.exp(-I1-I2)  # relative to I0

    return np.array([tr.Ebeam, tr.U['A2'], r_loc, Itot, ne_loc, Te_loc, lam,
                     sigmaEff_loc, I1, I2, L1, L2])


# %%
def fMaxwell(v, T, m):
    ''' Maxwelian distribution
    v in [m/s]
    T in [eV]
    '''
    if T < 0.01:
        return 0
    else:
        return ((m/(2*np.pi*T*1.6e-19))**1.5)*4*np.pi*v*v*np.exp(-m*v*v/(2*T*1.6e-19))  # T in [eV]


def genMaxwell(vtarget, Ttarget, m_target, vbeam, m_beam):
    ''' generalized Maxwellian distribution
            Ttarget in [eV]
    '''
    Ttarget = Ttarget*1.6e-19  # go to [J]
#    v = abs(vtarget-vbeam)
    v = vbeam-vtarget
    M = m_target*m_beam/(m_beam + m_target)
    return ((M/(2*np.pi*Ttarget))**0.5) * \
        (np.exp(-M*((v-vbeam)**2)/(2*Ttarget)) -
         np.exp(-M*((v+vbeam)**2)/(2*Ttarget))) * (v/vbeam)


def dSigmaEff(vtarget, Ttarget, m_target, sigma, vbeam, m_beam):
    ''' function calculates d(effective cross section) for monoenergetic
        beam and target gas
        Ttarget in [eV]
        sigma is a function of T in [eV]
    '''
    v = abs(vtarget-vbeam)
    try:
        sigmaEff = genMaxwell(vtarget, Ttarget, m_target, vbeam, m_beam) * \
             v * sigma((0.5*m_target*v**2)/1.6e-19)
    except ValueError:
        sigmaEff = genMaxwell(vtarget, Ttarget, m_target, vbeam, m_beam) * \
             v * 0.0
    return sigmaEff


# %%
if __name__ == '__main__':

    # plt.close('all')

    # %%
    kB = 1.38064852e-23  # Boltzman [J/K]
    m_e = 9.10938356e-31  # electron mass [kg]
    m_p = 1.6726219e-27  # proton mass [kg]

    geom = geomT15
    r_plasma = 0.65  # geom.r_plasma
    elon = geom.elon
    R = 1.45  # geom.R

    Btor = 1.0  # [T]
    Ipl = 1.0  # [MA]

    ne0 = 8.0  #5.0  # 1.5  # 15  # [x10^19 m-3]
    Te0 = 6.0  #2.0  # 1.0  # 15.0  # [keV]

    # %% import trajectories
    # tr_list = copy.deepcopy(traj_list_passed)
    # tr_list = copy.deepcopy(traj_list_a3b3)
#    filename = 'B{}_I{}//E80-320_UA2-20-80_alpha30_beta0_x250y-20z0.pkl'.format(str(int(Btor)), str(int(Ipl)))
    # filename = 'B1_I1/E100-300_UA26-33_alpha34.0_beta-10.0_x260y0z1.pkl'
    filename = 'B1_I1/E100-340_UA23-33_alpha34.0_beta-10.0_x260y-10z1.pkl'
    tr_list = hb.read_traj_list(filename, dirname='output')

    # %% LOAD IONIZATION RATES
    if tr_list[0].m/m_p > 200.:
        ion = 'Tl'
    else:
        ion = 'Cs'

    # <sigma*v> for Ion+ + e -> Ion2+
    filename = 'D:\\NRCKI\\Cross_sections\\' + ion + '\\rate' + ion + \
        '+_e_' + ion + '2+.txt'
    sigmaV12_e = np.loadtxt(filename)  # [0] Te [eV] [1] <sigma*v> [m^3/s]

    # <sigma*v> for Ion2+ + e -> Ion3+
    filename = 'D:\\NRCKI\\Cross_sections\\' + ion + '\\rate' + ion + \
        '2+_e_' + ion + '3+.txt'
    sigmaV23_e = np.loadtxt(filename)  # [0] Te [eV] [1] <sigma*v> [m^3/s]

    # %% interpolate rates
    sigmaEff12_e_interp = interpolate.interp1d(sigmaV12_e[:, 0]/1e3,
                                               sigmaV12_e[:, 1],
                                               kind='linear')  # Te in [keV]
    sigmaEff23_e_interp = interpolate.interp1d(sigmaV23_e[:, 0]/1e3,
                                               sigmaV23_e[:, 1],
                                               kind='linear')  # Te in [keV]

    # %%
    Itot = np.zeros([0, 12])
    for tr in tr_list:
        pass
        I_integrated = integrate_traj(tr, ne0, Te0, sigmaEff12_e_interp,
                                      sigmaEff23_e_interp)
        Itot = np.vstack([Itot, I_integrated[np.newaxis, :]])

    # get A2 and E lists
    Elist = np.array([tr.Ebeam for tr in tr_list])
    Elist = np.unique(Elist)
    A2list = np.array([tr.U['A2'] for tr in tr_list])
    A2list = np.unique(A2list)

    # %% plot results
    # plot scan
    Eb = 240.0  # beam energy [keV]
    fig, ax1 = plt.subplots()

    hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')

    # plot geometry
    geom.plot(ax1, axes='XY', plot_sep=False)

    A2list1 = []

    for tr in tr_list:
        if tr.Ebeam == Eb:
            A2list1.append(tr.U['A2'])
            # plot primary
            tr.plot_prim(ax1, axes='XY', color='k', full_primary=False)
            # plot secondary
            tr.plot_sec(ax1, axes='XY', color='r')

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list1))
    UA2_min = np.amin(np.array(A2list1))

    ax1.set_title('Ebeam={} keV, UA2:[{}, {}] kV, Btor = {} T, Ipl = {} MA'
                  .format(E, UA2_min,  UA2_max, Btor, Ipl))

    # plot plasma elipse
    x_plus = np.arange(0.0, r_plasma+0.01, 0.01)
    y = np.c_[-elon*np.sqrt((r_plasma**2-x_plus**2) / egg_fun(x_plus)),
              elon*np.sqrt((r_plasma**2-x_plus**2) / egg_fun(x_plus))]
    plt.plot(1.5+np.c_[x_plus, x_plus], y, color='m', linestyle='-')

    x_minus = np.arange(-r_plasma, 0.0, 0.01)
    y = np.c_[-elon*np.sqrt((r_plasma**2-x_minus**2) / egg_fun(x_minus)),
              elon*np.sqrt((r_plasma**2-x_minus**2) / egg_fun(x_minus))]
    plt.plot(1.5+np.c_[x_minus, x_minus], y, color='m', linestyle='-')

    # %% plot grid of attenuation
    fig, ax1 = plt.subplots()
    hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')

    # plot geometry
    geom.plot(ax1, axes='XY', plot_sep=True)

    N_A2 = A2list.shape[0]
    N_E = Elist.shape[0]

    A2_grid = np.full((N_E, 3, N_A2), np.nan)

    # find UA2 max and min
    UA2_max = np.amax(np.array(A2list))
    UA2_min = np.amin(np.array(A2list))
    # set title
    ax1.set_title('Eb = [{}, {}] keV, UA2 = [{}, {}] kV,'
                  ' Btor = {} T, Ipl = {} MA'
                  .format(tr_list[0].Ebeam, tr_list[-1].Ebeam, UA2_min,
                          UA2_max, Btor, Ipl))

    linestyle_E = '-'
    marker_E = 'p'
    E_grid = np.full((Itot.shape[0], 3), np.nan)
    c = Itot[:, 3]  # set color as Itot/I0
    k = -1
    # make a grid of constant E
    for i_E in range(0, N_E, 1):
        mask = (abs(Itot[:, 0] - Elist[i_E]) < 0.01)
        for tr in tr_list:
            if abs(tr.Ebeam - Elist[i_E]) < 0.1:
                k += 1
                # take the 1-st point of secondary trajectory
                x = tr.RV_sec[0, 0]
                y = tr.RV_sec[0, 1]
                z = tr.RV_sec[0, 2]
                E_grid[k, :] = [x, y, z]

    sc = ax1.scatter(E_grid[:, 0], E_grid[:, 1], s=80,
                     linestyle=linestyle_E,
                     # norm=colors.LogNorm(vmin=c.min(), vmax=c.max()),
                     norm=colors.LogNorm(vmin=1e-5, vmax=c.max()),
                     c=c,
                     cmap='jet',
                     marker=marker_E)
    plt.colorbar(sc, label=r'$I_{det} / I_0$')

    # %% plot grid of angles
    angles = np.full((Itot.shape[0], 2), np.nan)
    k = -1
    for tr in tr_list:
        k += 1
        angles[k, :] = hb.calc_angles(tr.RV_sec[-1, 3:])

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True)
    # plot geometry
    geom.plot(ax1, axes='XY', plot_sep=True)
    geom.plot(ax2, axes='XY', plot_sep=True)

    hbplot.set_axes_param(ax1, 'X (m)', 'Y (m)')
    hbplot.set_axes_param(ax2, 'X (m)', 'Y (m)')

    # plot grid with alpha coloring
    sc = ax1.scatter(E_grid[:, 0], E_grid[:, 1], s=80,
                     linestyle=linestyle_E,
                     c=angles[:, 0],
                     cmap='jet',
                     marker=marker_E)
    plt.colorbar(sc, ax=ax1, label=r'$\alpha (deg)$')

    # plot grid with beta coloring
    sc = ax2.scatter(E_grid[:, 0], E_grid[:, 1], s=80,
                     linestyle=linestyle_E,
                     c=angles[:, 1],
                     cmap='jet',
                     marker=marker_E)
    plt.colorbar(sc, ax=ax2, label=r'$\beta (deg)$')

    # %% plot ne and Te profiles
    fig, axs = plt.subplots(1, 2, sharex=True)
    rho = np.arange(0, 1.01, 0.01)
    ne_avg = round(integrate.simps(ne(rho, ne0), rho), 1)

    axs[0].plot(rho, Te(rho, Te0))
    axs[0].set_ylabel(r'$\ T_e (keV)$')

    axs[1].plot(rho, ne(rho, ne0))
    axs[1].set_ylabel(r'$\ n_e  (x10^{19} m^{-3})$')
    axs[1].set_title(r'$\ \barn_e =$' + str(ne_avg) + r'$\ x 10^{19} m^{-3}$')

    # format axes
    for ax in fig.get_axes():
        ax.set_xlabel(r'$\rho$')
        ax.set_xlim(0, 1.0)
        ax.grid()

    # %% plot Idet/I0
    plt.figure()
    for Eb in Elist:
        mask = (abs(Itot[:, 0] - Eb) < 0.01)
        plt.semilogy(Itot[mask, 2], Itot[mask, 3], 'o',
                     label='E={}'.format(Eb))
    plt.xlabel(r'$\rho_{SV}$')
    plt.ylabel(r'$\ I_{det} / I_0 $')
    plt.grid()
    plt.legend()

    # %% plot Idet/I0
    plt.figure()
    for Eb in Elist:
        mask = (abs(Itot[:, 0] - Eb) < 0.01)
        plt.semilogy(Itot[mask, 1], Itot[mask, 3], '-o',
                     label='E={}'.format(Eb))
    plt.xlabel('UA2 (kV)')
    plt.ylabel(r'$\ I_{det} / I_0 $')
    plt.grid()
    plt.legend()

    # %% plot atten factor
    plt.figure()
    for Eb in Elist:
        mask = (abs(Itot[:, 0] - Eb) < 0.01)
        plt.semilogy(Itot[mask, 2], np.exp(-1*Itot[mask, 8]-1*Itot[mask, 9]),
                     '-o', label='E={}'.format(Eb))
    plt.ylabel(r'Atten. factor ($e^{-R_1-R_2}$)')
    plt.xlabel(r'$\rho_{SV}$')
    plt.grid()
    plt.legend()

    # %% plot rates
    plt.figure()
    plt.semilogx(sigmaV12_e[:, 0], sigmaV12_e[:, 1]*1e6, 'o', color='k',
                 label=r'$Tl^+$+e $\rightarrow$ $Tl^{2+}$+2e')
    Temp = np.linspace(min(sigmaV12_e[:, 0]), max(sigmaV12_e[:, 0]), num=10000)
    plt.semilogx(Temp, sigmaEff12_e_interp(Temp/1e3)*1e6, '-', color='k')

    plt.semilogx(sigmaV23_e[:, 0], sigmaV23_e[:, 1]*1e6, '^', color='k',
                 label=r'$Tl^{2+}$+e $\rightarrow$ $Tl^{3+}$+2e')
    Temp = np.linspace(min(sigmaV23_e[:, 0]), max(sigmaV23_e[:, 0]), num=40000)
    plt.semilogx(Temp, sigmaEff23_e_interp(Temp/1e3)*1e6, '--', color='k')

    plt.xlabel(r'$E_{e}$ (eV)')
    plt.ylabel(r'<$\sigma$V> ($cm^3/s$)')
    plt.grid(linestyle='--', which='both')
    leg = plt.legend()
    for artist, text in zip(leg.legendHandles, leg.get_texts()):
        col = artist.get_color()
        if isinstance(col, np.ndarray):
            col = col[0]
        text.set_color(col)

    plt.show()
