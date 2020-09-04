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

def run_UO2_Fe():
    print('### test UO2-Fe coherent interface ###\n')
    # tdb filepath
    tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    # components
    comps = ['O', 'U', 'FE']
    # mass density laws (from Barrachin2004)
    constituentDensityLaws = {
        'U1'   : lambda T: 17270.0-1.358*(T-1408),
        'ZR1'  : lambda T: 6844.51-0.609898*T+2.05008E-4*T**2-4.47829E-8*T**3+3.26469E-12*T**4,
        'O2U1' : lambda T: 8860.0-9.285E-1*(T-3120),
        'O2ZR1': lambda T: 5150-0.445*(T-2983),
        'FE1'  : lambda T: 7030 - 0.88*(T-1808),
        'NI1'  : lambda T: 7900 - 1.19*(T-1728),
        'CR1'  : lambda T: 6290 - 0.72*(T-2178),
        'O1'   : lambda T: 1.141, # set to meaningless value but ok as, no 'free' oxygen in the considered mixtures
        'FE1O1' : lambda T: 7030 - 0.88*(T-1808), # set to Fe value but ok as, almost no such component in the considered mixtures
        'FE1O1_5' : lambda T: 7030 - 0.88*(T-1808), # set to Fe value but ok as, almost no such component in the considered mixtures
    }
    # phase names
    phasenames = ['LIQUID', 'LIQUID']
    # pressure
    P = 1E5
    # Given initial alloy composition. x0 is the mole fractions of U, Fe.
    x0 = [0.25, 1.0-0.25-0.49]
    # Composition range for searching initial interfacial equilibrium composition.
    limit = [0.0001, 0.9]
    # Composition step for searching initial interfacial equilibrium composition.
    dx = 0.1

    T = 3300.0
    # Molar volumes of pure components evaluated at x0 and kept constant afterwards
    CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
    model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
    functions=model.constantPartialMolarVolumeFunctions(x0, constituentDensityLaws, 1E-5)
    # calculate global equilibrium
    model = CoherentGibbsEnergy_OC(T, P, phasenames, False)
    mueq = model.chemicalpotential(x0)

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
            enforceGridMinimizerForLocalEq=False,
            mueq=mueq
        )
        print('at T=', T, ' sigma=', sigma.Interfacial_Energy.values, '\n')
    else:
        print('at T=', T, ' out of the miscibility gap')

def run():
    print('### test U-O-Zr-Fe coherent interface in the liquid miscibility gap ###\n')
    # tdb filepath
    tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    # components
    comps = ['O', 'U', 'ZR', 'FE']
    # mass density laws (from Barrachin2004)
    constituentDensityLaws = {
        'U1'   : lambda T: 17270.0-1.358*(T-1408),
        'ZR1'  : lambda T: 6844.51-0.609898*T+2.05008E-4*T**2-4.47829E-8*T**3+3.26469E-12*T**4,
        'O2U1' : lambda T: 8860.0-9.285E-1*(T-3120),
        'O2ZR1': lambda T: 5150-0.445*(T-2983),
        'FE1'  : lambda T: 7030 - 0.88*(T-1808),
        'NI1'  : lambda T: 7900 - 1.19*(T-1728),
        'CR1'  : lambda T: 6290 - 0.72*(T-2178),
        'O1'   : lambda T: 1.141, # set to meaningless value but ok as, no 'free' oxygen in the considered mixtures
        'FE1O1'  : lambda T: 7030 - 0.88*(T-1808), # set to Fe value but ok as, almost no such component in the considered mixtures
        'FE1O1_5'  : lambda T: 7030 - 0.88*(T-1808), # set to Fe value but ok as, almost no such component in the considered mixtures
    }
    # phase names
    phasenames = ['LIQUID', 'LIQUID']
    # pressure
    P = 1E5
    # Given initial alloy composition. x0 is the mole fractions of U, Zr, Fe.
    # RU/Zr=1.2 CZr=0.3 xSteel=0.1
    x0 = [0.20131833168321586, 0.1677652764026799, 0.12762056270606442]
    # Composition step for searching initial interfacial equilibrium composition.
    dx = 0.1

    # temperature range
    Tmin = 2800.0
    Tmax = 4200.0
    Trange = np.linspace(Tmin, Tmax, num=10, endpoint=True)
    results = pd.DataFrame(columns=['temperature', 'n_phase1', 'n_phase2', 'xU_phase1', 'xU_phase2', 'xU_interface', 'sigma'])

    for T in Trange:
        # Molar volumes of pure components evaluated at x0 and kept constant afterwards
        CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
        model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
        functions=model.constantPartialMolarVolumeFunctions(x0, constituentDensityLaws, 1E-5)

        # calculate global equilibrium and retrieve associated chemical potentials
        model = CoherentGibbsEnergy_OC(T, 1E5, phasenames)
        mueq = model.chemicalpotential(x0)
        phasesAtEquilibrium = oc.getPhasesAtEquilibrium()
        phasesAtEquilibriumMolarAmounts = phasesAtEquilibrium.getPhaseMolarAmounts()
        if (len(phasesAtEquilibriumMolarAmounts)==1):
            # it is possible that the miscibility gap has not been detected correctly (can happen when T increases)
            #print(phasesAtEquilibriumMolarAmounts)
            # ad hoc strategy: 1) calculate an equilibrium at lower temperature (hopefully finding the two phases)
            #                  2) redo the calculation at the target temperature afterwards without the grid minimizer
            model = CoherentGibbsEnergy_OC(T-300.0, 1E5, phasenames)
            mueq = model.chemicalpotential(x0)
            phasesAtEquilibrium = oc.getPhasesAtEquilibrium()
            phasesAtEquilibriumMolarAmounts = phasesAtEquilibrium.getPhaseMolarAmounts()
            #print(phasesAtEquilibriumMolarAmounts)
            oc.setTemperature(T)
            oc.calculateEquilibrium(gmStat.Off)
            mueq = model.getChemicalPotentials()
            phasesAtEquilibrium = oc.getPhasesAtEquilibrium()
            phasesAtEquilibriumMolarAmounts = phasesAtEquilibrium.getPhaseMolarAmounts()

        phasesAtEquilibriumElementCompositions = phasesAtEquilibrium.getPhaseElementComposition()
        print(phasesAtEquilibriumMolarAmounts)
        if (set(phasesAtEquilibriumMolarAmounts)==set(['LIQUID#1', 'LIQUID_AUTO#2'])):
            # Composition range for searching initial interfacial equilibrium composition
            # calculated from the actual phase compositions
            componentsWithLimits = comps[1:]
            limit = [ [1.0, 0.0] for each in componentsWithLimits ]
            for phase in phasesAtEquilibriumElementCompositions:
                for element in phasesAtEquilibriumElementCompositions[phase]:
                    elementMolarFraction = phasesAtEquilibriumElementCompositions[phase][element]
                    if element in componentsWithLimits:
                        limit[componentsWithLimits.index(element)][0] = min(limit[componentsWithLimits.index(element)][0], elementMolarFraction)
                        limit[componentsWithLimits.index(element)][1] = max(limit[componentsWithLimits.index(element)][1], elementMolarFraction)
            limit = [ [each[0]+dx, each[1]-dx] for each in limit ]
            print('limits: ', limit)
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
                enforceGridMinimizerForLocalEq=False,
                mueq=mueq
            )
            print('at T=', T, ' sigma=', sigma.Interfacial_Energy.values, '\n')
            if (np.abs(sigma.Interfacial_Energy.values)>1E-6):
                # store results in pandas dataframe
                results = results.append({'temperature' : T,
                                        'n_phase1' : phasesAtEquilibriumMolarAmounts['LIQUID#1'],
                                        'n_phase2' : phasesAtEquilibriumMolarAmounts['LIQUID_AUTO#2'],
                                        'xU_phase1' : phasesAtEquilibriumElementCompositions['LIQUID#1']['U'],
                                        'xU_phase2' : phasesAtEquilibriumElementCompositions['LIQUID_AUTO#2']['U'],
                                        'xZr_phase1' : phasesAtEquilibriumElementCompositions['LIQUID#1']['ZR'],
                                        'xZr_phase2' : phasesAtEquilibriumElementCompositions['LIQUID_AUTO#2']['ZR'],
                                        'xFe_phase1' : phasesAtEquilibriumElementCompositions['LIQUID#1']['FE'],
                                        'xFe_phase2' : phasesAtEquilibriumElementCompositions['LIQUID_AUTO#2']['FE'],
                                        'xU_interface' : sigma.Interfacial_Composition.values[1],
                                        'xZr_interface' : sigma.Interfacial_Composition.values[2],
                                        'xFe_interface' : sigma.Interfacial_Composition.values[3],
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
    results.to_csv('macro_liquidMG_UOZrFe_run.csv')

def fit():
    results = pd.read_csv('macro_liquidMG_UOZrFe_run.csv')
    # Function to calculate the power-law with constants sigma0, Tc, mu, sigmaC
    def power_law_plus_const(T, sigma0, Tc, mu, sigmaC):
        return sigma0*np.power(1.0-T/Tc, mu)+sigmaC
    def power_law_no_const(T, sigma0, Tc, mu):
        return sigma0*np.power(1.0-T/Tc, mu)
    # Fit the power-law data
    power_law = power_law_no_const
    print(results['temperature'])
    print(results['sigma'])
    pars, cov = curve_fit(f=power_law, xdata=results['temperature'], ydata=results['sigma'], p0=[0.7, results['temperature'][len(results['temperature']) - 1], 1.9], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
    print(power_law(results['temperature'], *pars))
    res = results['sigma'] - power_law(results['temperature'], *pars)
    print(pars, stdevs)

    plt.rcParams['figure.figsize'] = (12,7)
    fig,axes=plt.subplots(1,2,constrained_layout=True)
    # Plots associated with interfacial energy
    ax = axes[0]
    ax.grid(True)
    ax.plot(results['temperature'], results['sigma'], marker = 'o', ls='', color='tab:cyan', label='calculated values: $\sigma_{calculated}$')
    legLabel = 'fit: $\sigma_{fit}='+'{0:4.3f} (1-T/{1:4.1f})^'.format(pars[0], pars[1])+'{'+'{0:4.3f}'.format(pars[2])+'}$'
    ax.plot(results['temperature'], power_law(results['temperature'], *pars), linestyle='--', linewidth=2, color='black', label=legLabel)
    ax.set_xlabel('temperature T (K)',fontsize=12)
    ax.set_ylabel('interfacial energy $\sigma$ (N.m$^{-1}$)',fontsize=12)
    ax.legend(loc='upper right')
    ax = axes[1]
    ax.grid(True)
    ax.plot(results['temperature'], res, marker = 'o', ls='', color='tab:cyan')
    ax.set_xlabel('temperature T (K)',fontsize=12)
    ax.set_ylabel('fit residuals $\sigma_{fit} - \sigma_{calculated}$ (N.m$^{-1}$)',fontsize=12)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    plt.savefig('macro_liquidMG_UOZrFe_fit.pdf')
    plt.show()

if __name__ == '__main__':
    run_UO2_Fe()
    run()
    fit()
