import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import os
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC
from openiec.calculate.calcsigma_OC import SigmaCoherent_OC2
from pyOC import opencalphad as oc
from pyOC import GridMinimizerStatus as gmStat
from scipy.optimize import minimize, Bounds, LinearConstraint, NonlinearConstraint, BFGS
from functools import partial

def constituentToEndmembersConverter(constituentMolarFractions, constituentsDescription):
    endmemberMolarFractions = {
        'U1'   : constituentMolarFractions['sublattice 0']['U+4']*constituentMolarFractions['sublattice 1']['VA'],
        'O2U1' : constituentMolarFractions['sublattice 0']['U+4']*constituentMolarFractions['sublattice 1']['O-2'],
        'O1'   : constituentMolarFractions['sublattice 1']['O'],
        'ZR1'  : constituentMolarFractions['sublattice 0']['ZR+4']*constituentMolarFractions['sublattice 1']['VA'],
        'FE1'  : constituentMolarFractions['sublattice 0']['FE+2']*constituentMolarFractions['sublattice 1']['VA'],
        'O2ZR1' : constituentMolarFractions['sublattice 0']['ZR+4']*constituentMolarFractions['sublattice 1']['O-2'],
        'FE1O1' : constituentMolarFractions['sublattice 0']['FE+2']*constituentMolarFractions['sublattice 1']['O-2'],
        'FE1O1_5' : constituentMolarFractions['sublattice 1']['FEO3/2'],
    }
    endmemberMolarMasses = {
        'U1'   : constituentsDescription['U+4']['mass'],
        'O1'   : constituentsDescription['O']['mass'],
        'O2U1' : constituentsDescription['U+4']['mass']+2.0*constituentsDescription['O']['mass'],
        'ZR1'   : constituentsDescription['ZR+4']['mass'],
        'FE1'   : constituentsDescription['FE+2']['mass'],
        'O2ZR1' : constituentsDescription['ZR+4']['mass']+2.0*constituentsDescription['O']['mass'],
        'FE1O1' : constituentsDescription['FE+2']['mass']+1.0*constituentsDescription['O']['mass'],
        'FE1O1_5' : constituentsDescription['FE+2']['mass']+1.5*constituentsDescription['O']['mass'],
    }
    endMemberMassFractions = {k : endmemberMolarFractions[k]*endmemberMolarMasses[k] for k in endmemberMolarFractions}
    factor=1.0/sum(endMemberMassFractions.values())
    for k in endMemberMassFractions:
        endMemberMassFractions[k] = endMemberMassFractions[k]*factor
    return endMemberMassFractions

def ComputeEquilibriumWithConstraints(objectfunction, x0, bulkX, method="trust-constr", tol=1e-14):
    print("********************")
    print("starting point: ", x0)
    print("objective function value at starting point: ", objectfunction(x0))
    print("excluded points (bulk composition): ", bulkX)
    print(bulkX)
    linearConstraint = LinearConstraint([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], keep_feasible=True)
    n = len(bulkX)
    def cons_f(x):
        f0=np.sqrt((x[0]-bulkX[0][0])**2+(x[1]-bulkX[1][0])**2+(x[2]-bulkX[2][0])**2)
        f1=np.sqrt((x[0]-bulkX[0][1])**2+(x[1]-bulkX[1][1])**2+(x[2]-bulkX[2][1])**2)
        return [f0, f1]
    def cons_J(x):
        f = cons_f(x)
        return [ [(x[0]-bulkX[0][0])/f[0], (x[1]-bulkX[1][0])/f[0], (x[2]-bulkX[2][0])/f[0]],
                 [(x[0]-bulkX[0][1])/f[1], (x[1]-bulkX[1][1])/f[1], (x[2]-bulkX[2][1])/f[1]] ]
    def cons_H(x, v):
        f = cons_f(x)
        a11 = 1/f[0]-(x[0]-bulkX[0][0])**2/f[0]**3
        a12 = -(x[0]-bulkX[0][0])*(x[1]-bulkX[1][0])/f[0]**3
        a13 = -(x[0]-bulkX[0][0])*(x[2]-bulkX[2][0])/f[0]**3
        a22 = 1/f[0]-(x[1]-bulkX[1][0])**2/f[0]**3
        a23 = -(x[1]-bulkX[1][0])*(x[2]-bulkX[2][0])/f[0]**3
        a33 = 1/f[0]-(x[2]-bulkX[2][0])**2/f[0]**3
        b11 = 1/f[1]-(x[0]-bulkX[0][1])**2/f[1]**3
        b12 = -(x[0]-bulkX[0][1])*(x[1]-bulkX[1][1])/f[1]**3
        b13 = -(x[0]-bulkX[0][1])*(x[2]-bulkX[2][1])/f[1]**3
        b22 = 1/f[1]-(x[1]-bulkX[1][1])**2/f[1]**3
        b23 = -(x[1]-bulkX[1][1])*(x[2]-bulkX[2][1])/f[1]**3
        b33 = 1/f[1]-(x[2]-bulkX[2][1])**2/f[1]**3
        return v[0]*np.array([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]]) + v[1]*np.array([[b11, b12, b13], [b12, b22, b23], [b13, b23, b33]])
    nonlinearConstraint = NonlinearConstraint(cons_f, 1E-6, np.inf, jac=cons_J, hess=cons_H, keep_feasible=True)
    res = minimize(objectfunction, x0, method=method, constraints=[linearConstraint, nonlinearConstraint],
    options={'xtol': tol, 'gtol': tol, 'maxiter': 3000, 'initial_constr_penalty': 0.5, 'verbose': 1})
    print(res.x)
    print(res.fun)
    #if (res.fun>1E-2):
    #    raise ValueError('misconvergence!')
    print("********************")
    return res.x

def run():
    print('### test U-O-Zr-Fe coherent interface in the liquid miscibility gap ###\n')
    # tdb filepath
    tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    # tdbFile='tests/TAF_uzrofe_V10.TDB'
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
    # RU/Zr=0.60 CZr=0.3 xSteel=0.1
    x0 = [0.1550142, 0.2583569, 0.1215864]
    # Composition step for searching initial interfacial equilibrium composition.
    #dx = 0.5
    # Convergence criterion for loop on interfacial composition
    epsilonX = 1E-5

    # temperature range
    Tmin = 2900.0
    Tmax = 4200.0
    Trange = np.linspace(Tmin, Tmax, num=11, endpoint=True)
    results = pd.DataFrame(columns=['temperature', 'n_phase1', 'n_phase2', 'xU_phase1', 'xU_phase2','xZr_phase1', 'xZr_phase2', 'xFe_phase1', 'xFe_phase2','xU_interface','xZr_interface','xFe_interface', 'sigma','VmU','VmZr','VmFe'])

    x=None
    for T in Trange:
            # calculate global equilibrium and retrieve associated chemical potentials
            CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
            oc.raw().pytqtgsw(4) # no merging of grid points
            #oc.raw().pytqtgsw(23) # denser grid
            model = CoherentGibbsEnergy_OC(T, 1E5, phasenames)
            mueq = model.chemicalpotential(x0)
            phasesAtEquilibrium = oc.getPhasesAtEquilibrium()
            phasesAtEquilibriumMolarAmounts = phasesAtEquilibrium.getPhaseMolarAmounts()
            if (len(phasesAtEquilibriumMolarAmounts)==1):
                # it is possible that the miscibility gap has not been detected correctly (can happen when T increases)
                #print(phasesAtEquilibriumMolarAmounts)
                # ad hoc strategy: 1) calculate an equilibrium at lower temperature (hopefully finding the two phases)
                #                  2) redo the calculation at the target temperature afterwards without the grid minimizer
                model = CoherentGibbsEnergy_OC(Tmin, 1E5, phasenames)
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
            print(phasesAtEquilibriumElementCompositions)
            if (set(phasesAtEquilibriumMolarAmounts)==set(['LIQUID#1', 'LIQUID_AUTO#2'])):
                # Composition range for searching initial interfacial equilibrium composition
                # calculated from the actual phase compositions
                componentsWithLimits = comps[1:]
                #limit = [ [1.0, 0.0] for each in componentsWithLimits ]
                #for phase in phasesAtEquilibriumElementCompositions:
                #    for element in phasesAtEquilibriumElementCompositions[phase]:
                #        elementMolarFraction = phasesAtEquilibriumElementCompositions[phase][element]
                #        if element in componentsWithLimits:
                #            limit[componentsWithLimits.index(element)][0] = min(limit[componentsWithLimits.index(element)][0], elementMolarFraction)
                #            limit[componentsWithLimits.index(element)][1] = max(limit[componentsWithLimits.index(element)][1], elementMolarFraction)
                #limit = [ [each[0]+dx*(each[1]-each[0]), each[1]-dx*(each[1]-each[0])] for each in limit ]
                bulkX = [ [ phasesAtEquilibriumElementCompositions[phase][element] for phase in phasesAtEquilibriumMolarAmounts ] for element in componentsWithLimits ]

                notConverged = True
                if (x==None):
                    x = [ 0.5*(phasesAtEquilibriumElementCompositions['LIQUID#1'][comp] + phasesAtEquilibriumElementCompositions['LIQUID_AUTO#2'][comp]) for comp in componentsWithLimits ]
                # Iterate on interfacial molar composition
                while (notConverged):
                    # Molar volumes of pure components evaluated at x
                    CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
                    model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
                    if ('TAF' in tdbFile):
                        functions=model.constantPartialMolarVolumeFunctions(x, constituentDensityLaws, 1E-5, constituentToEndmembersConverter)
                    else:
                        functions=model.constantPartialMolarVolumeFunctions(x, constituentDensityLaws, 1E-5)
                    # calculate interfacial energy
                    sigma = SigmaCoherent_OC2(
                        T=T,
                        x0=x0,
                        db=tdbFile,
                        comps=comps,
                        phasenames=phasenames,
                        purevms=functions,
                        guess=x,
                        computeEquilibriumFunction=partial(ComputeEquilibriumWithConstraints, bulkX=bulkX),
                        enforceGridMinimizerForLocalEq=False,
                        mueq=mueq
                        )
                    print('at T=', T, ' sigma=', sigma.Interfacial_Energy.values, '\n')
                    notConverged = np.linalg.norm(x[:]-sigma.Interfacial_Composition.values[1:], np.inf)>epsilonX
                    print('convergence: ', not notConverged, x[:], sigma.Interfacial_Composition.values[1:])
                    x[:]=sigma.Interfacial_Composition.values[1:]
                # store results in pandas dataframe
                if (np.abs(sigma.Interfacial_Energy.values)>1E-6):
                    print(sigma, "\n")
                    if (abs(np.max(sigma.Partial_Interfacial_Energy.values)-np.min(sigma.Partial_Interfacial_Energy.values))>1E-4):
                        raise ValueError('wrong value discarded')
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
                                              'VmU' : functions[1](T),
                                              'VmZr' : functions[2](T),
                                              'VmFe' : functions[3](T),
                                              'VmO' : functions[0](T),
                                              },
                            ignore_index = True)
                else:
                    print(sigma, "\n")
                    raise ValueError('wrong value discarded')
            else:
                print('at T=', T, ' out of the miscibility gap')
            print('phases at equilibrium:', phasesAtEquilibriumMolarAmounts)
    # write csv result file
    results.to_csv('macro_liquidMG_UOZrFe_run.csv')

def run2():
    print('### test U-O coherent interface in the liquid miscibility gap ###\n')
    # tdb filepath
    #tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    #tdbFile=os.environ['TDBDATA_PRIVATE']+'/NUCLEA-17_1_mod.TDB'
    #tdbFile=os.environ['TDBDATA_PRIVATE']+'/NUCLEA-19_1_mod.TDB'
    tdbFile='tests/TAF_uzrofe_V10.TDB'
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

    constituentDensityLaws['U'] = constituentDensityLaws['U1']
    constituentDensityLaws['ZR'] = constituentDensityLaws['ZR1']
    constituentDensityLaws['O'] = constituentDensityLaws['O1']
    constituentDensityLaws['FE'] = constituentDensityLaws['FE1']

    # phase names
    phasenames = ['LIQUID', 'LIQUID']
    # pressure
    P = 1E5
    # Given initial alloy composition. x0 is the mole fractions of U, Zr, Fe.
    # RU/Zr=0.60 CZr=0.3 xSteel=0.1
    x0 = [0.1550142, 0.2583569, 0.1215864]
    # Composition step for searching initial interfacial equilibrium composition.
    #dx = 0.5
    # Convergence criterion for loop on interfacial composition
    epsilonX = 1E-5

    inputs = pd.read_csv('macro_liquidMG_UOZrFe_run.csv')
    results = pd.DataFrame(columns=['temperature', 'n_phase1', 'n_phase2', 'xU_phase1', 'xU_phase2','xZr_phase1', 'xZr_phase2', 'xFe_phase1', 'xFe_phase2','xU_interface','xZr_interface','xFe_interface', 'VmU', 'VmZr','VmFe','sigma'])

    x = None
    for i,T in enumerate(inputs['temperature']):
            # calculate global equilibrium and retrieve associated chemical potentials
            CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
            oc.raw().pytqtgsw(4) # no merging of grid points
            #oc.raw().pytqtgsw(23) # denser grid
            model = CoherentGibbsEnergy_OC(T, 1E5, phasenames)
            mueq = model.chemicalpotential(x0)
            phasesAtEquilibrium = oc.getPhasesAtEquilibrium()
            phasesAtEquilibriumMolarAmounts = phasesAtEquilibrium.getPhaseMolarAmounts()
            if (len(phasesAtEquilibriumMolarAmounts)==1):
                # it is possible that the miscibility gap has not been detected correctly (can happen when T increases)
                #print(phasesAtEquilibriumMolarAmounts)
                # ad hoc strategy: 1) calculate an equilibrium at lower temperature (hopefully finding the two phases)
                #                  2) redo the calculation at the target temperature afterwards without the grid minimizer
                model = CoherentGibbsEnergy_OC(2800.0, 1E5, phasenames)
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
            print(phasesAtEquilibriumElementCompositions)
            if (set(phasesAtEquilibriumMolarAmounts)==set(['LIQUID#1', 'LIQUID_AUTO#2'])):
                # Composition range for searching initial interfacial equilibrium composition
                # calculated from the actual phase compositions
                componentsWithLimits = comps[1:]
                #limit = [ [1.0, 0.0] for each in componentsWithLimits ]
                #for phase in phasesAtEquilibriumElementCompositions:
                #    for element in phasesAtEquilibriumElementCompositions[phase]:
                #        elementMolarFraction = phasesAtEquilibriumElementCompositions[phase][element]
                #        if element in componentsWithLimits:
                #            limit[componentsWithLimits.index(element)][0] = min(limit[componentsWithLimits.index(element)][0], elementMolarFraction)
                #            limit[componentsWithLimits.index(element)][1] = max(limit[componentsWithLimits.index(element)][1], elementMolarFraction)
                #limit = [ [each[0]+dx*(each[1]-each[0]), each[1]-dx*(each[1]-each[0])] for each in limit ]
                bulkX = [ [ phasesAtEquilibriumElementCompositions[phase][element] for phase in phasesAtEquilibriumMolarAmounts ] for element in componentsWithLimits ]

                if (x==None):
                    x = [ 0.5*(phasesAtEquilibriumElementCompositions['LIQUID#1'][comp] + phasesAtEquilibriumElementCompositions['LIQUID_AUTO#2'][comp]) for comp in componentsWithLimits ]
                #x = x0.copy()
                # Molar volumes of pure components evaluated at x
                functions = [ lambda _: inputs['VmO'][i], lambda _: inputs['VmU'][i], lambda _: inputs['VmZr'][i], lambda _: inputs['VmFe'][i]]
                # calculate interfacial energy
                sigma = SigmaCoherent_OC2(
                    T=T,
                    x0=x0,
                    db=tdbFile,
                    comps=comps,
                    phasenames=phasenames,
                    purevms=functions,
                    guess=x,
                    computeEquilibriumFunction=partial(ComputeEquilibriumWithConstraints, bulkX=bulkX),
                    enforceGridMinimizerForLocalEq=False,
                    mueq=mueq
                )
                print('at T=', T, ' sigma=', sigma.Interfacial_Energy.values, '\n')
                x[:]=sigma.Interfacial_Composition.values[1:]
                # Store result
                if (np.abs(sigma.Interfacial_Energy.values)>1E-6):
                    # store results in pandas dataframe
                    print(sigma, "\n")
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
                                              'VmU' : functions[0](T),
                                              'VmZr' : functions[1](T),
                                              'VmFe' : functions[2](T),
                                            },
                            ignore_index = True)
                else:
                    print(sigma, "\n")
                    raise ValueError('wrong value discarded')
            else:
                print('at T=', T, ' out of the miscibility gap')
            print('phases at equilibrium:', phasesAtEquilibriumMolarAmounts)
    # write csv result file
    results.to_csv('macro_liquidMG_UOZrFe_run2.csv')

def run3(tdbFile, RUZr):
    print('### test U-O-Zr-Fe coherent interface in the liquid miscibility gap ###\n')
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
    # initial alloy compositions. x0 is the mole fractions of U, Zr, Fe.
    read = pd.read_csv('tests/{0:2.1f}RUZr.csv'.format(RUZr), delim_whitespace=True)
    # Composition step for searching initial interfacial equilibrium composition.
    #dx = 0.5
    # Convergence criterion for loop on interfacial composition
    epsilonX = 1E-4

    # temperature range
    T = 3000
    # Trange = np.linspace(Tmin, Tmax, num=10, endpoint=True)
    results = pd.DataFrame(columns=['temperature', 'n_phase1', 'n_phase2', 'xU_phase1', 'xU_phase2','xZr_phase1', 'xZr_phase2', 'xFe_phase1', 'xFe_phase2','xU_interface','xZr_interface','xFe_interface', 'sigma','VmU','VmZr','VmFe'])

    x = None
    for ii in range(read.shape[0]):
            x0=[read['xU'][ii],read['xZr'][ii],read['xFe'][ii]]
            print("*********({0:d}/{1:d})*********".format(ii+1, read.shape[0]))
            print("x0: ",x0)
            # calculate global equilibrium and retrieve associated chemical potentials
            CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
            oc.raw().pytqtgsw(4) # no merging of grid points
            #oc.raw().pytqtgsw(23) # denser grid
            model = CoherentGibbsEnergy_OC(T, 1E5, phasenames)
            mueq = model.chemicalpotential(x0)
            phasesAtEquilibrium = oc.getPhasesAtEquilibrium()
            phasesAtEquilibriumMolarAmounts = phasesAtEquilibrium.getPhaseMolarAmounts()
            if (len(phasesAtEquilibriumMolarAmounts)==1):
                # it is possible that the miscibility gap has not been detected correctly (can happen when T increases)
                #print(phasesAtEquilibriumMolarAmounts)
                # ad hoc strategy: 1) calculate an equilibrium at lower temperature (hopefully finding the two phases)
                #                  2) redo the calculation at the target temperature afterwards without the grid minimizer
                model = CoherentGibbsEnergy_OC(2900, 1E5, phasenames)
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
                #limit = [ [1.0, 0.0] for each in componentsWithLimits ]
                #for phase in phasesAtEquilibriumElementCompositions:
                #for element in phasesAtEquilibriumElementCompositions[phase]:
                #        elementMolarFraction = phasesAtEquilibriumElementCompositions[phase][element]
                #        if element in componentsWithLimits:
                #            limit[componentsWithLimits.index(element)][0] = min(limit[componentsWithLimits.index(element)][0], elementMolarFraction)
                #            limit[componentsWithLimits.index(element)][1] = max(limit[componentsWithLimits.index(element)][1], elementMolarFraction)
                #limit = [ [each[0]+dx*(each[1]-each[0]), each[1]-dx*(each[1]-each[0])] for each in limit ]
                bulkX = [ [ phasesAtEquilibriumElementCompositions[phase][element] for phase in phasesAtEquilibriumMolarAmounts ] for element in componentsWithLimits ]

                notConverged = True
                if (x==None):
                    x = [ 0.5*(phasesAtEquilibriumElementCompositions['LIQUID#1'][comp] + phasesAtEquilibriumElementCompositions['LIQUID_AUTO#2'][comp]) for comp in componentsWithLimits ]
                # Iterate on interfacial molar composition
                while (notConverged):
                    # Molar volumes of pure components evaluated at x
                    CoherentGibbsEnergy_OC.initOC(tdbFile, comps)
                    model = CoherentGibbsEnergy_OC(T, P, phasenames[0], False)
                    if ('TAF' in tdbFile):
                        functions=model.constantPartialMolarVolumeFunctions(x, constituentDensityLaws, 1E-5, constituentToEndmembersConverter)
                    else:
                        functions=model.constantPartialMolarVolumeFunctions(x, constituentDensityLaws, 1E-5)
                    # calculate interfacial energy
                    sigma = SigmaCoherent_OC2(
                        T=T,
                        x0=x0,
                        db=tdbFile,
                        comps=comps,
                        phasenames=phasenames,
                        purevms=functions,
                        guess=x,
                        computeEquilibriumFunction=partial(ComputeEquilibriumWithConstraints, bulkX=bulkX),
                        enforceGridMinimizerForLocalEq=False,
                        mueq=mueq
                    )
                    print('at T=', T, ' sigma=', sigma.Interfacial_Energy.values, '\n')
                    notConverged = np.linalg.norm(x[:]-sigma.Interfacial_Composition.values[1:], np.inf)>epsilonX
                    print('convergence: ', not notConverged, x[:], sigma.Interfacial_Composition.values[1:])
                    x[:]=sigma.Interfacial_Composition.values[1:]
                # store results in pandas dataframe
                if (np.abs(sigma.Interfacial_Energy.values)>1E-5):
                    print(sigma, "\n")
                    if (abs(np.max(sigma.Partial_Interfacial_Energy.values)-np.min(sigma.Partial_Interfacial_Energy.values))>1E-4):
                        print(np.min(sigma.Partial_Interfacial_Energy.values))
                        print(np.max(sigma.Partial_Interfacial_Energy.values))
                        raise ValueError('wrong value discarded')
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
                                              'VmU' : functions[1](T),
                                              'VmZr' : functions[2](T),
                                              'VmFe' : functions[3](T),
                                              'VmO' : functions[0](T),
                                              },
                            ignore_index = True)
                else:
                    raise ValueError('wrong value discarded')
            else:
                print('at T=', T, ' out of the miscibility gap')
            print('phases at equilibrium:', phasesAtEquilibriumMolarAmounts)
    # write csv result file
    if ('TAF' in tdbFile):
        results.to_csv('macro_liquidMG_UOZrFe_run3_TAFID_RUZR={0:2.1f}.csv'.format(RUZr))
    else:
        results.to_csv('macro_liquidMG_UOZrFe_run3_RUZR={0:2.1f}.csv'.format(RUZr))

def fit():
    results = pd.read_csv('macro_liquidMG_UOZrFe_run2.csv')
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

    # Plots associated with composition
    ax = axes[0,1]
    ax.grid(True)
    ax.plot(results['xU_interface'], results['sigma'], marker = 'o', ls='--', color='tab:cyan')
    ax.set_ylabel('interfacial energy $\sigma$ (N.m$^{-1}$)',fontsize=12)
    ax.set_xlabel('interface U molar fraction',fontsize=12)
    ax = axes[1,0]
    ax.grid(True)
    ax.plot(results['xZr_interface'], results['sigma'], marker = 'o', ls='--', color='tab:cyan')
    ax.set_ylabel('interfacial energy $\sigma$ (N.m$^{-1}$)',fontsize=12)
    ax.set_xlabel('interface Zr molar fraction',fontsize=12)
    ax = axes[1,1]
    ax.grid(True)
    ax.plot(results['xFe_interface'], results['sigma'], marker = 'o', ls='--', color='tab:cyan')
    ax.set_ylabel('interfacial energy $\sigma$ (N.m$^{-1}$)',fontsize=12)
    ax.set_xlabel('interface Fe molar fraction',fontsize=12)

    plt.savefig('macro_liquidMG_UOZrFe_fit.pdf')
    plt.show()

def plot(tdbFile, RUZr):
    inputs = pd.read_csv('tests/{0:2.1f}RUZr.csv'.format(RUZr), delim_whitespace=True)
    CZr=inputs['CZr']
    xSteel=inputs['xSteel']
    # write csv result file
    if ('TAF' in tdbFile):
        results = pd.read_csv('macro_liquidMG_UOZrFe_run3_TAFID_RUZR={0:2.1f}.csv'.format(RUZr))
    else:
        results = pd.read_csv('macro_liquidMG_UOZrFe_run3_RUZR={0:2.1f}.csv'.format(RUZr))
    #
    epsilon=1E-4
    def calculateSet(array, tol):
        sortedArray = array.copy()
        sortedArray.sort()
        results = [sortedArray.pop(0), ]
        for value in sortedArray:
            if abs(results[-1] - value) <= tol:
                continue
            results.append(value)
        return results
    #
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:brown', 'tab:pink']
    markers=['x', '+', 'o', '*', '^', 'v', '<', '>']
    prop_cycle = cycler(color=colors) + cycler(marker=markers) + cycler(markevery=[0.1]*len(markers))
    prop_cycle2 = prop_cycle * cycler(linestyle=['-', '--', '-.'])
    #
    ftSize = 10
    plt.rcParams["figure.figsize"] = (12,10)
    plt.rcParams["legend.fontsize"] = ftSize
    CZr_set = calculateSet(inputs['CZr'].tolist(), epsilon)
    x_Fe_set = calculateSet(inputs['xSteel'].tolist(), epsilon)
    #
    fig,axes=plt.subplots(2,2,constrained_layout=True)
    ax1 = axes[0,0]
    ax1.grid(True)
    ax1.set_prop_cycle(prop_cycle)
    ax2 = axes[0,1]
    ax2.grid(True)
    ax2.set_prop_cycle(prop_cycle2)
    ax3 = axes[1,0]
    ax3.grid(True)
    ax3.set_prop_cycle(prop_cycle2)
    ax4 = axes[1,1]
    ax4.grid(True)
    ax4.set_prop_cycle(prop_cycle2)
    for valCZr in CZr_set:
        indI = [i for i, val in enumerate(CZr) if abs(val - valCZr) < epsilon]
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)
        csf = ax1.plot(xSteel[indI], results['sigma'][indI], label=legLabel)
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - interfacial liquid"
        csf = ax2.plot(xSteel[indI], results['xU_interface'][indI], label=legLabel)
        xmin = [max(results['xU_phase1'][i], results['xU_phase2'][i]) for i in indI]
        xmax = [min(results['xU_phase1'][i], results['xU_phase2'][i]) for i in indI]
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - bulk metal"
        csf = ax2.plot(xSteel[indI], xmin, label=legLabel)
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - bulk oxide"
        csf = ax2.plot(xSteel[indI], xmax, label=legLabel)
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - interfacial liquid"
        csf = ax3.plot(xSteel[indI], results['xZr_interface'][indI], label=legLabel)
        xmin = [max(results['xZr_phase1'][i], results['xZr_phase2'][i]) for i in indI]
        xmax = [min(results['xZr_phase1'][i], results['xZr_phase2'][i]) for i in indI]
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - bulk metal"
        csf = ax3.plot(xSteel[indI], xmin, label=legLabel)
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - bulk oxide"
        csf = ax3.plot(xSteel[indI], xmax, label=legLabel)
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - interfacial liquid"
        csf = ax4.plot(xSteel[indI], results['xFe_interface'][indI], label=legLabel)
        xmin = [max(results['xFe_phase1'][i], results['xFe_phase2'][i]) for i in indI]
        xmax = [min(results['xFe_phase1'][i], results['xFe_phase2'][i]) for i in indI]
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - bulk metal"
        csf = ax4.plot(xSteel[indI], xmax, label=legLabel)
        legLabel="$C_{Zr}"+"={0:3.2f}$".format(valCZr)+" - bulk oxide"
        csf = ax4.plot(xSteel[indI], xmin, label=legLabel)
    ax1.set_xlabel("$x_{steel}$", fontsize=ftSize)
    ax1.set_ylabel("interfacial energy $\sigma$ (N.m$^{-1}$)", fontsize=ftSize)
    ax1.set_title("$R_{U/Zr}"+"={0:2.1f}$".format(RUZr), fontsize=ftSize)
    #ax1.legend(loc="best", ncol=2)
    ax2.set_xlabel("$x_{steel}$", fontsize=ftSize)
    ax2.set_ylabel("U molar fraction", fontsize=ftSize)
    ax2.set_title("$R_{U/Zr}"+"={0:2.1f}$".format(RUZr), fontsize=ftSize)
    #ax2.legend(loc="best", ncol=2)
    ax3.set_xlabel("$x_{steel}$", fontsize=ftSize)
    ax3.set_ylabel("Zr molar fraction", fontsize=ftSize)
    ax3.set_title("$R_{U/Zr}"+"={0:2.1f}$".format(RUZr), fontsize=ftSize)
    #ax3.legend(loc="best", ncol=2)
    ax4.set_xlabel("$x_{steel}$", fontsize=ftSize)
    ax4.set_ylabel("Fe molar fraction", fontsize=ftSize)
    ax4.set_title("$R_{U/Zr}"+"={0:2.1f}$".format(RUZr), fontsize=ftSize)
    #ax4.legend(loc="best", ncol=2)

    lines, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(lines, labels, loc = 'center')

    plt.savefig('macro_liquidMG_UOZrFe_plot.pdf')
    plt.show()


if __name__ == '__main__':
    #run()
    #run2()
    #fit()
    #
    # tdb filepath
    #tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
    tdbFile='tests/TAF_uzrofe_V10.TDB'
    RUZr=1.0
    run3(tdbFile, RUZr)
    plot(tdbFile, RUZr)
