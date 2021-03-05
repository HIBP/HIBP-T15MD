import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import interpolate
import hibplib as hb
import hibpplotlib as hbplot

#%%
# Calculate HIBP beam attenuation in T-15MD tokamak
def get_rho(x, y, z, r_pl=0.65, elon=1.7, R=1.5):
    '''
    function returns normalized radius
    R - major radius
    elon - elongation along y
    r_pl - plasma radius along x
    '''
    x1 = np.sqrt(x**2 + z**2)
    return np.sqrt((x1-R)**2 + (y/elon)**2) / r_pl
    
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
    # first of all add the first point of secondary traj to the primary traj
    # find the distances
    distances = np.array([np.linalg.norm(tr.RV_Prim[i,0:3] - tr.RV_Sec[0,0:3]) 
                    for i in range(tr.RV_Prim.shape[0])])
    sorted_indices = np.argsort(distances)
    # find position where to insert the new point
    index_to_insert = max(sorted_indices[0:2])
    tr.RV_Prim = np.insert(tr.RV_Prim, index_to_insert, tr.RV_Sec[0,:], axis=0)
    
    # integrals over primary and secondary trajectory
    I1 = 0.0
    I2 = 0.0
    L1 = 0.0
    L2 = 0.0
    # integration loop
    # integrating primary trajectory
    for i in range(1, index_to_insert+1):
        x1 = tr.RV_Prim[i-1,0]
        y1 = tr.RV_Prim[i-1,1]
        z1 = tr.RV_Prim[i-1,2]
        x2 = tr.RV_Prim[i,0]
        y2 = tr.RV_Prim[i,1]
        z2 = tr.RV_Prim[i,2]
        
        rho1 = get_rho(x1, y1, z1)
        rho2 = get_rho(x2, y2, z2)
        
        if (rho1 <= 1)&(rho2 <= 1):
            dl = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
            r_loc = 0.5 * (rho1 + rho2)
            ne_loc = 1e19 * ne(r_loc, ne0) 
            Te_loc = Te(r_loc, Te0)
            I1 += dl*sigmaEff12(Te_loc)*ne_loc
            L1 += dl
            
    # integrating secondary trajectory
    for i in range(1, tr.RV_Sec.shape[0]):
        x1 = tr.RV_Sec[i-1,0]
        y1 = tr.RV_Sec[i-1,1]
        z1 = tr.RV_Sec[i-1,2]
        x2 = tr.RV_Sec[i,0]
        y2 = tr.RV_Sec[i,1]
        z2 = tr.RV_Sec[i,2]
        
        rho1 = get_rho(x1, y1, z1)
        rho2 = get_rho(x2, y2, z2)
        
        if (rho1 <= 1)&(rho2 <= 1):
            dl = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
            r_loc = 0.5 * (rho1 + rho2)
            ne_loc = 1e19 * ne(r_loc, ne0) 
            Te_loc = Te(r_loc, Te0)
            I2 += dl*sigmaEff23(Te_loc)*ne_loc
            L2 += dl
            
    r_loc = get_rho(tr.RV_Sec[0,0], tr.RV_Sec[0,1], tr.RV_Sec[0,2])
    if r_loc <= 0.99:
        Te_loc = Te(r_loc, Te0)
        ne_loc = 1e19 * ne(r_loc, ne0) 
        sigmaEff_loc = sigmaEff12(Te_loc)
    else:
        Te_loc = 0.
        ne_loc = 0.
        sigmaEff_loc = 0.
                
    # calculate total value with integrals
    lam = 0.005 # [m]
    I = 2*ne_loc*sigmaEff_loc*lam*math.exp(-I1-I2)

    return np.array([tr.Ebeam, tr.UA2, r_loc, I, ne_loc, Te_loc, lam, 
                     sigmaEff_loc, I1, I2, L1, L2])

#%%
def fMaxwell(v, T, m): 
    ''' Maxwelian distribution
    v in [m/s]
    T in [eV]
    '''
    if T < 0.01:
        return 0
    else:
        return ((m/(2*np.pi*T*1.6e-19))**1.5)*4*np.pi*v*v*np.exp(-m*v*v/(2*T*1.6e-19)) # T in [eV]

def genMaxwell(vtarget, Ttarget, m_target, vbeam, m_beam):
    ''' generalized Maxwellian distribution
            Ttarget in [eV]
    '''    
    Ttarget = Ttarget*1.6e-19 # go to [J]
#    v = abs(vtarget-vbeam)
    v = vbeam-vtarget
    M = m_target*m_beam/(m_beam + m_target)
    return ((M/(2*np.pi*Ttarget))**0.5) * \
            (np.exp(-M*((v-vbeam)**2)/(2*Ttarget)) - \
             np.exp(-M*((v+vbeam)**2)/(2*Ttarget))) * (v/vbeam)
    
def dSigmaEff(vtarget, Ttarget, m_target, sigma, vbeam, m_beam):
    ''' function calculates d(effective cross section) for monoenergetic beam and target gas
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

#%%
if __name__ == '__main__':
    
    plt.close('all')
    
    # %%
    ion = 'Tl'
    
    if ion == 'Tl':
        m_ion = 204*1.6605e-27 # Tl mass [kg]
    elif ion == 'Cs':
        m_ion = 133*1.6605e-27 # Cs mass [kg]
    
    kB = 1.38064852e-23 # Boltzman [J/K]
    m_e = 9.10938356e-31 # electron mass [kg]
    m_p = 1.6726219e-27 # proton mass [kg]
    E = 240.0 # beam energy [keV]
    v0 = math.sqrt(2*E*1.6e-16 / m_ion) # initial particle velocity [m/s]    
    
    Btor = 1.0 # [T]
    Ipl = 1.0 # [MA]
    
    ne0 = 1.5 # [x10^19 m-3]
    Te0 = 1.0 # [keV]
    
    #%% LOAD IONIZATION RATES
    # <sigma*v> for Ion+ + e -> Ion2+ 
    filename = 'D:\\NRCKI\\Cross_sections\\' + ion + '\\rate' + ion+ '+_e_' +ion + '2+.txt'
    sigmaV12_e = np.loadtxt(filename) # [0] Te [eV] [1] <sigma*v> [m^3/s]
        
    # <sigma*v> for Ion2+ + e -> Ion3+ 
    filename = 'D:\\NRCKI\\Cross_sections\\' + ion + '\\rate' + ion + '2+_e_' + ion + '3+.txt'
    sigmaV23_e = np.loadtxt(filename) # [0] Te [eV] [1] <sigma*v> [m^3/s]
    
    #%% interpolate rates
    sigmaEff12_e_interp = interpolate.interp1d(sigmaV12_e[:,0]/1e3, sigmaV12_e[:,1]/v0, 
                                               kind='linear') # Te in [keV]
    sigmaEff23_e_interp = interpolate.interp1d(sigmaV23_e[:,0]/1e3, sigmaV23_e[:,1]/v0,
                                               kind='linear') # Te in [keV]
    
    #%% plot rates
    plt.figure()
    plt.semilogx(sigmaV12_e[:,0], sigmaV12_e[:,1]*1e6, 'o', color='k', 
                 label=r'$Tl^+$+e $\rightarrow$ $Tl^{2+}$+2e')
    Temp = np.linspace(min(sigmaV12_e[:,0]), max(sigmaV12_e[:,0]), num=10000)
    plt.semilogx(Temp, sigmaEff12_e_interp(Temp/1e3)*1e6*v0, '-', color='k')
    
    plt.semilogx(sigmaV23_e[:,0], sigmaV23_e[:,1]*1e6, '^', color='k', 
                 label=r'$Tl^{2+}$+e $\rightarrow$ $Tl^{3+}$+2e')
    Temp = np.linspace(min(sigmaV23_e[:,0]), max(sigmaV23_e[:,0]), num=40000)
    plt.semilogx(Temp, sigmaEff23_e_interp(Temp/1e3)*1e6*v0, '--', color='k')
        
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
    
    #%%
    # import trajectories
#    filename = 'B{}_I{}//E80-320_UA2-20-80_alpha30_beta0_x250y-20z0.pkl'.format(str(int(Btor)), str(int(Ipl)))
#    traj_list = hb.ReadTrajList(filename, dirname='output')
    
    Itot = np.zeros([0,12])
    for tr in traj_list:
        if abs(tr.Ebeam - E) < 0.1:
            I_integrated = integrate_traj(tr, ne0, Te0, sigmaEff12_e_interp, sigmaEff23_e_interp)
            Itot = np.vstack([Itot, I_integrated[np.newaxis, :]])
            
    #%% plot results
    # plot scan
    plot_scan_xy(traj_list, np.array([[0,0,0]]), Geometry(), Ebeam, Btor, Ipl)
    # plot plasma elipse
    r = 0.65
    R = 1.5
    elon = 1.7
    x_plus = np.arange(0.0, r+0.01, 0.01)
    y = np.c_[-elon*np.sqrt(r**2-x_plus**2), elon*np.sqrt(r**2-x_plus**2)]
    plt.plot(1.5+np.c_[x_plus, x_plus], y, color='m', linestyle='--')
    
    x_minus = np.arange(-r, 0.0, 0.01)
    y = np.c_[-elon*np.sqrt(r**2-x_minus**2), elon*np.sqrt(r**2-x_minus**2)]
    plt.plot(1.5+np.c_[x_minus, x_minus], y, color='m', linestyle='--')
    
    #%% plot ne and Te profiles
    fig, axs = plt.subplots(1, 2, sharex=True)
    rho = np.arange(0, 1.01, 0.01)
    axs[0].plot(rho, Te(rho, Te0))
    axs[0].set_ylabel(r'$\ T_e (keV)$')
    
    axs[1].plot(rho, ne(rho, ne0))
    axs[1].set_ylabel(r'$\ n_e  (x10^{19} m^{-3})$')
    
    # format axes
    for ax in fig.get_axes():
        ax.set_xlabel(r'$\rho$')
        ax.set_xlim(0, 1.0)
        ax.grid()
    
    # plot Idet/I0
    plt.figure()
    plt.semilogy(Itot[:,2], Itot[:,3], 'o')
    plt.xlabel(r'$\rho_{SV}$')
    plt.ylabel(r'$\ I_{det} / I_0 $')
    plt.grid()
    
    #%%
    plt.figure()
    plt.semilogy(Itot[:,1], Itot[:,3], 'o')
    plt.xlabel('UA2 (kV)')
    plt.ylabel(r'$\ I_{det} / I_0 $')
    plt.grid()
    
    #%% plot atten factor
    plt.figure()
    plt.semilogy(Itot[:,2], np.exp(-1*Itot[:,8]-1*Itot[:,9]), 'o')
    plt.ylabel(r'Atten. factor ($e^{-R_1-R_2}$)')
    plt.xlabel(r'$\rho_{SV}$')
    plt.grid()
    