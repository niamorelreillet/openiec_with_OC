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
from pyOC import GridMinimizerStatus as gmStat

def constituentToEndmembersConverter(constituentMolarFractions, constituentsDescription):
    endmemberMolarFractions = {
        'O2U1' : constituentMolarFractions['sublattice 0']['U+4']*constituentMolarFractions['sublattice 1']['O-2'],
        'U1'   : constituentMolarFractions['sublattice 0']['U+4']*constituentMolarFractions['sublattice 1']['VA'],
        'O1'   : constituentMolarFractions['sublattice 1']['O']
    }
    endmemberMolarMasses = {
        'U1'   : constituentsDescription['U+4']['mass'],
        'O1'   : constituentsDescription['O']['mass'],
        'O2U1' : constituentsDescription['U+4']['mass']+2.0*constituentsDescription['O']['mass']
    }
    endMemberMassFractions = {k : endmemberMolarFractions[k]*endmemberMolarMasses[k] for k in endmemberMolarFractions}
    factor=1.0/sum(endMemberMassFractions.values())
    for k in endMemberMassFractions:
        endMemberMassFractions[k] = endMemberMassFractions[k]*factor
    return endMemberMassFractions

def run():
    print('### test U-O coherent interface in the liquid miscibility gap ###\n')
    # tdb filepath
    #tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    #tdbFile=os.environ['TDBDATA_PRIVATE']+'/NUCLEA-17_1_mod.TDB'
    tdbFile='tests/TAF_uzrofe_V10.TDB'
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
    constituentDensityLaws['U'] = constituentDensityLaws['U1']
    constituentDensityLaws['ZR'] = constituentDensityLaws['ZR1']
    constituentDensityLaws['O'] = constituentDensityLaws['O1']

    # phase names
    phasenames = ['LIQUID', 'LIQUID']
    # pressure
    P = 1E5
    # Given initial alloy composition. x0 is the mole fraction of U.
    x0 = [0.65]
    # Composition step for searching initial interfacial equilibrium composition.
    dx = 0.05

    # temperature range
    Tmin = 2800.0
    Tmax = 4400.0
    Trange = np.linspace(Tmin, Tmax, num=60, endpoint=True)
    results = pd.DataFrame(columns=['temperature', 'n_phase1', 'n_phase2', 'xU_phase1', 'xU_phase2', 'xU_interface', 'sigma'])

    for T in Trange:
        # Molar volumes of pure components evaluated at x0 and kept constant afterwards
        CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
        model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
        functions=model.constantPartialMolarVolumeFunctions(x0, constituentDensityLaws, 1E-5, constituentToEndmembersConverter)
        #functions=model.constantPartialMolarVolumeFunctions(x0, constituentDensityLaws, 1E-5)

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
                                        'xU_interface' : sigma.Interfacial_Composition.values[1],
                                        'sigma' : sigma.Interfacial_Energy.values,
                                        },
                        ignore_index = True)
            else:
                raise ValueError('wrong value discarded')
        else:
            print('at T=', T, ' out of the miscibility gap')
        print('phases at equilibrium:', phasesAtEquilibriumMolarAmounts)
    # write csv result file
    results.to_csv('macro_liquidMG_UO_run.csv')

def fit():
    results = pd.read_csv('macro_liquidMG_UO_run.csv')
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
    fig,axes=plt.subplots(2,2,constrained_layout=True)
    # Plots associated with interfacial energy
    ax = axes[0,0]
    ax.grid(True)
    ax.plot(results['temperature'], results['sigma'], marker = 'o', ls='', color='tab:cyan', label='calculated values: $\sigma_{calculated}$')
    legLabel = 'fit: $\sigma_{fit}='+'{0:4.3f} (1-T/{1:4.1f})^'.format(pars[0], pars[1])+'{'+'{0:4.3f}'.format(pars[2])+'}$'
    ax.plot(results['temperature'], power_law(results['temperature'], *pars), linestyle='--', linewidth=2, color='black', label=legLabel)
    ax.set_xlabel('temperature T (K)',fontsize=12)
    ax.set_ylabel('interfacial energy $\sigma$ (N.m$^{-1}$)',fontsize=12)
    ax.legend(loc='upper right')
    ax = axes[0,1]
    ax.grid(True)
    ax.plot(results['temperature'], res, marker = 'o', ls='', color='tab:cyan')
    ax.set_xlabel('temperature T (K)',fontsize=12)
    ax.set_ylabel('fit residuals $\sigma_{fit} - \sigma_{calculated}$ (N.m$^{-1}$)',fontsize=12)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # Plots associated with composition
    ax = axes[1,0]
    ax.plot(results['xU_phase1'], results['temperature'], marker = '', ls='-', color='tab:red', label='bulk liquid 1')
    ax.plot(results['xU_phase2'], results['temperature'], marker = '', ls='-', color='tab:green', label='bulk liquid 2')
    ax.plot(results['xU_interface'], results['temperature'], marker = '', ls='-', color='tab:cyan', label='interface')
    ax.set_ylabel('temperature T (K)',fontsize=12)
    ax.set_xlabel('U molar fraction',fontsize=12)
    ax.legend(loc='upper right')
    ax = axes[1,1]
    ax.plot(results['xU_interface'], results['sigma'], marker = 'o', ls='--', color='tab:cyan')
    ax.set_ylabel('interfacial energy $\sigma$ (N.m$^{-1}$)',fontsize=12)
    ax.set_xlabel('interface U molar fraction',fontsize=12)

    plt.savefig('macro_liquidMG_UO_fit.pdf')
    plt.show()

if __name__ == '__main__':
    run()
    fit()
