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

def run_NUCLEA():
    print('### test U-O coherent interface in the liquid miscibility gap ###\n')
    # tdb filepath
    tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    # tdbFile=os.environ['TDBDATA_PRIVATE']+'/NUCLEA-17_1_mod.TDB'
#    tdbFile='tests/TAF_uzrofe_V10.TDB'
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
    phasenames = ['LIQUID#1', 'LIQUID#2']
    # pressure & temp
    P = 1E5
    T = 3200
    # Given initial alloy composition. x0 is the mole fraction of U.
    x0min = [0.5]
    x0max = [0.7]
    x0range = np.linspace(x0min[0],x0max[0],num=20, endpoint=True)
    
    # Composition step for searching initial interfacial equilibrium composition.
    dx = 0.05
    results = pd.DataFrame(columns=['X_U', 'n_phase1', 'n_phase2', 'mu_U', 'mu_O'])

    for x0 in x0range:
        # Molar volumes of pure components evaluated at x0 and kept constant afterwards
        CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
        model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
        # # functions=model.constantPartialMolarVolumeFunctions(x0, constituentDensityLaws, 1E-5, constituentToEndmembersConverter)
        functions=model.constantPartialMolarVolumeFunctions([x0], constituentDensityLaws, 1E-5)

        # calculate global equilibrium and retrieve associated chemical potentials
        model = CoherentGibbsEnergy_OC(T, 1E5, phasenames)
        mueq = model.chemicalpotential([x0])
        print('mu_U= ',mueq[1])
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
            # store results in pandas dataframe
            results = results.append({'X_U': x0,
                                      'n_phase1' : phasesAtEquilibriumMolarAmounts['LIQUID#1'],
                                      'n_phase2' : phasesAtEquilibriumMolarAmounts['LIQUID_AUTO#2'],
                                      'mu_U': mueq[1],
                                      'mu_O': mueq[0],
                                        },
                        ignore_index = True)
    results.to_csv('UO_muvsx_NUCLEA.csv')
    
def constituentToEndmembersConverter(constituentMolarFractions, constituentsDescription):
    endmemberMolarFractions = {
        'O2U1' : constituentMolarFractions['sublattice 0']['U+4']*constituentMolarFractions['sublattice 1']['O-2'],
        'U1'   : constituentMolarFractions['sublattice 0']['U+4']*constituentMolarFractions['sublattice 1']['VA'],
        'O1'   : constituentMolarFractions['sublattice 1']['O']
    }
    endmemberMolarMasses = {
        'U1'   : constituentsDescription['U+4']['mass'],
        'O1'   : constituentsDescription['O']['mass'],
        'O2U1' : constituentsDescription['U+4']['mass']+2.0*constituentsDescription['O-2']['mass']
    }
    endMemberMassFractions = {k : endmemberMolarFractions[k]*endmemberMolarMasses[k] for k in endmemberMolarFractions}
    factor=1.0/sum(endMemberMassFractions.values())
    for k in endMemberMassFractions:
        endMemberMassFractions[k] = endMemberMassFractions[k]*factor
    return endMemberMassFractions
    
def run_TAFID():
    print('### test U-O coherent interface in the liquid miscibility gap ###\n')
    # tdb filepath
    tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    # tdbFile=os.environ['TDBDATA_PRIVATE']+'/NUCLEA-17_1_mod.TDB'
#    tdbFile='tests/TAF_uzrofe_V10.TDB'
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
    phasenames = ['LIQUID#1', 'LIQUID#2']
    # pressure & temp
    P = 1E5
    T = 3200
    # Given initial alloy composition. x0 is the mole fraction of U.
    x0min = [0.5]
    x0max = [0.7]
    x0range = np.linspace(x0min[0],x0max[0],num=20, endpoint=True)
    
    # Composition step for searching initial interfacial equilibrium composition.
    dx = 0.05
    results_1 = pd.DataFrame(columns=['X_U', 'n_phase1', 'n_phase2', 'mu_U', 'mu_O'])

    for x0 in x0range:
        # Molar volumes of pure components evaluated at x0 and kept constant afterwards
        CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
        model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
        # # functions=model.constantPartialMolarVolumeFunctions(x0, constituentDensityLaws, 1E-5, constituentToEndmembersConverter)
        functions=model.constantPartialMolarVolumeFunctions([x0], constituentDensityLaws, 1E-5)

        # calculate global equilibrium and retrieve associated chemical potentials
        model = CoherentGibbsEnergy_OC(T, 1E5, phasenames)
        mueq = model.chemicalpotential([x0])
        print('mu_U= ',mueq[1])
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
            # store results in pandas dataframe
            results_1 = results_1.append({'X_U': x0,
                                      'n_phase1' : phasesAtEquilibriumMolarAmounts['LIQUID#1'],
                                      'n_phase2' : phasesAtEquilibriumMolarAmounts['LIQUID_AUTO#2'],
                                      'mu_U': mueq[1],
                                      'mu_O': mueq[0],
                                        },
                        ignore_index = True)
    results_1.to_csv('UO_muvsx_TAFID.csv')
    
def fit():
    results = pd.read_csv('UO_muvsx_NUCLEA.csv')
    results_1 = pd.read_csv('UO_muvsx_TAFID.csv')

    plt.rcParams['figure.figsize'] = (12,7)
    fig,axes=plt.subplots(2,2,constrained_layout=True)
    # Plots associated with interfacial energy
    ax = axes[0,0]
    ax.grid(True)
    # ax.plot(results['X_U'], results['mu_U'], marker = 'o', ls='', color='tab:cyan', label='calculated values: $\mu{O}$')
    ax.plot(np.array(results['X_U']).astype(np.float32), np.array(results['mu_U']).astype(np.float32), marker = 'o', ls='', color='tab:cyan', label='calculated values: $\mu{U}$')
    ax.plot(np.array(results_1['X_U']).astype(np.float32), np.array(results_1['mu_U']).astype(np.float32), marker = 'x', ls='', color='tab:blue', label='calculated values: $\mu{U}$')
    ax.set_xlabel('Composition of U',fontsize=12)
    ax.set_ylabel('Chemical potential $\mu$ (J.kg$^{-1}$)',fontsize=12)
    ax.legend(loc='upper right')

    ax = axes[0,1]
    ax.grid(True)
    ax.plot(np.array(results['X_O']).astype(np.float32), np.array(results['mu_O']).astype(np.float32), marker = 'o', ls='', color='tab:cyan', label='calculated values: $\mu{U}$')
    ax.plot(np.array(results_1['X_O']).astype(np.float32), np.array(results_1['mu_O']).astype(np.float32), marker = 'x', ls='', color='tab:blue', label='calculated values: $\mu{U}$')
    ax.set_xlabel('Composition of O',fontsize=12)
    ax.set_ylabel('Chemical potential $\mu$ (J.kg$^{-1}$)',fontsize=12)
    ax.legend(loc='upper right')
    plt.savefig('UO_MUvsX_fit.pdf')
    plt.show()

if __name__ == '__main__':
    run_NUCLEA()
    run_TAFID()
    fit()
