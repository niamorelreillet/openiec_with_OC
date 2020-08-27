import unittest
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC
from openiec.calculate.calcsigma_OC import SigmaCoherent_OC
from pyOC import opencalphad as oc

def testUO():
    print('### test U-O coherent interface in the liquid miscibility gap ###\n')
    # tdb filepath
    tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    # components
    comps = ['O', 'U']
    # mass density laws (from Barrachin2004)
    constituentDensityLaws = {
        'U1'   : lambda T: 17270.0-1.358*(T-1408),
        'ZR1'  : lambda T: 6844.51-0.609898*T+2.05008E-4*T**2-4.47829E-8*T**3+3.26469E-12*T**4,
        'O2U1' : lambda T: 8860.0-9.285E-1*(T-3120),
        'O2ZR1': lambda T: 5150-0.445*(T-2983),
        'O1'   : lambda T: 1.141 # set to meaningless value but ok as, no 'free' oxygen in the considered mixtures
    }
    # phase names
    phasenames = ['LIQUID', 'LIQUID']
    # pressure
    P = 1E5
    # Given initial alloy composition. x0 is the mole fraction of U.
    x0 = [0.9]
    # Composition range for searching initial interfacial equilibrium composition.
    limit = [0.0001, 0.9]
    # Composition step for searching initial interfacial equilibrium composition.
    dx = 0.1

    # temperature range
    Tmin = 2800.0
    Tmax = 3400.0
    Trange = np.linspace(Tmin, Tmax, num=50, endpoint=True)
    results = pd.DataFrame(columns=['temperature', 'n_phase1', 'n_phase2', 'xU_phase1', 'xU_phase2', 'xU_interface', 'sigma'])

    for T in Trange:
        # Molar volumes of pure components evaluated at x0 and kept constant afterwards
        CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
        model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
        functions=model.constantPartialMolarVolumeFunctions(x0, constituentDensityLaws, 1E-5)
        # calculate global equilibrium
        model = CoherentGibbsEnergy_OC(T, P, phasenames, False)
        #oc.raw().pytqtgsw(19) # set sparse grid for convergence issues
        model.chemicalpotential(x0, False)

        phasesAtEquilibrium = oc.getPhasesAtEquilibrium()
        phasesAtEquilibriumMolarAmounts = phasesAtEquilibrium.getPhaseMolarAmounts()
        diff = set(phasesAtEquilibriumMolarAmounts) - set(['LIQUID#1', 'LIQUID_AUTO#2'])
        if (len(diff)==0):
            phasesAtEquilibriumElementCompositions = phasesAtEquilibrium.getPhaseElementComposition()
            # calculate interfacial energy
            sigma = SigmaCoherent_OC(
                T=T,
                x0=x0,
                db=tdbFile,
                comps=comps,
                phasenames=phasenames,
                purevms=functions,
                limit=limit,
                dx=dx,
                enforceGridMinimizerForLocalEq=False
            )
            print('at T=', T, ' sigma=', sigma.Interfacial_Energy.values, '\n')
            if (sigma.Interfacial_Energy.values>0):
                # store results in pandas dataframe
                results = results.append({'temperature' : T,
                                        'n_phase1' : phasesAtEquilibriumMolarAmounts['LIQUID#1'],
                                        'n_phase2' : phasesAtEquilibriumMolarAmounts['LIQUID_AUTO#2'],
                                        'xU_phase1' : phasesAtEquilibriumElementCompositions['LIQUID#1']['U'],
                                        'xU_phase2' : phasesAtEquilibriumElementCompositions['LIQUID_AUTO#2']['U'],
                                        'xU_interface' : sigma.Interfacial_Composition.values[1],
                                        'sigma' : sigma.Interfacial_Energy.values,
                                        },
                        ignore_index = True)
            else:
                print('wrong value discarded')
        else:
            print('at T=', T, ' out of the miscibility gap')
        print('phases at equilibrium:', phasesAtEquilibriumMolarAmounts)
        print(phasesAtEquilibriumElementCompositions)
    # write csv result file
    results.to_csv('macro_liquidMG_UO_testUO.csv')

def fitUO():
    results = pd.read_csv('macro_liquidMG_UO_testUO.csv')
    # Function to calculate the power-law with constants sigma0, Tc, mu, sigmaC
    def power_law(T, sigma0, Tc, mu, sigmaC):
        return sigma0*np.power(1.0-T/Tc, mu)+sigmaC
    # Fit the power-law data
    print(results['temperature'])
    print(results['sigma'])
    pars, cov = curve_fit(f=power_law, xdata=results['temperature'], ydata=results['sigma'], p0=[0.2, results['temperature'][len(results['temperature']) - 1], 1.3, 0.0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    print(power_law(results['temperature'], *pars))
    res = results['sigma'] - power_law(results['temperature'], *pars)
    print(pars, stdevs)

    # Plot the fit data as an overlay on the scatter data
    plt.rcParams['figure.figsize'] = (12,7)
    fig,axes=plt.subplots(1,2,constrained_layout=True)
    ax = axes[0]
    ax.scatter(results['temperature'], results['sigma'], s=20, color='#00b3b3')
    ax.plot(results['temperature'], power_law(results['temperature'], *pars), linestyle='--', linewidth=2, color='black')
    #ax.set_xscale('log')
    #ax.set_xlim(1E-3,max(results['temperature']))
    #ax.set_yscale('log')
    #ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    #ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    ax = axes[1]
    ax.scatter(results['temperature'], res, s=20, color='#00b3b3')
    #ax.set_xscale('log')
    #ax.set_xlim(1E-3,max(results['temperature']))
    #ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    plt.show()

if __name__ == '__main__':
    testUO()
    fitUO()
