
from openiec.model.sigmacoint import SigmaCoherentInterface
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC
from openiec.model.sigmasolliq import SigmaPureMetal, SigmaSolidLiquidInterface
from openiec.property.solliqenergy import SolutionGibbsEnergy, InterfacialGibbsEnergy
from openiec.property.meltingenthalpy import MeltingEnthalpy as hm
from openiec.property.molarinfarea import MolarInterfacialArea
from openiec.property.molarvolume import MolarVolume, InterficialMolarVolume
from openiec.calculate.minimize import SearchEquilibrium, ComputeEquilibrium
from openiec.utils.decorafunc import wraptem
from pycalphad import Database
from pyOC import opencalphad as oc
import numpy as np
from xarray import Dataset

def SigmaCoherent_OC(
    T, x0, db, comps, phasenames, purevms, intervms=[], limit=[0, 1.0], dx=0.01, enforceGridMinimizerForLocalEq=False
):
    """
    Calculate the coherent interfacial energy in alloys.

    Parameters
    -----------
    T: float
        Given temperature.
    x0: list
        Initial alloy composition.
    db : Database
        Database containing the relevant parameters.
    comps : list
        Names of components to consider in the calculation.
    phasenames : list
        Names of phase model to build.
    limit: list
        The limit of composition for searching interfacial composition in equilibrium.
    purevms: list
        The molar volumes of pure components (expression) or the interfacial molar volumes (functions)
    dx: float
        The step of composition for searching interfacial composition in equilibrium.

    Returns:
    -----------
    Components：list of str
        Given components.
    Temperature: float
        Given temperature.
    Initial_Alloy_Composition: list
        Given initial alloy composition.
    Interfacial_Composition: list
        Interfacial composition of the grid minimization.
    Partial_Interfacial_Energies: list
        Partial interfacial energies of components.
    Interfacial_Energy: float
        Requested interfacial energies.

    Return type: xarray Dataset
    """
    if (type(purevms[0])==list):
        # the molar volumes of pure components are given as expressions (original openIEC implementation)
        phasevm = [MolarVolume(Database(db), phasenames[i], comps, purevms[i]) for i in range(2)]
        _vmis = InterficialMolarVolume(*phasevm)
        """decorate the _vmis to release the constains on temperature"""
        vmis = [wraptem(T, f) for f in _vmis]
    else:
        # the molar volumes of pure components directly given as functions
        vmis = purevms

    CoherentGibbsEnergy_OC.initOC(db, comps)
    model = CoherentGibbsEnergy_OC(T, 1E5, phasenames)

    """Chemical potentials in two-phase equilibrium"""
    #oc.raw().pytqtgsw(19) # set sparse grid for convergence issues
    mueq = model.chemicalpotential(x0)

    CoherentGibbsEnergy_OC.initOC(db, comps)

    """Chemical potentials in two bulk phases"""
    model_phase = [
        CoherentGibbsEnergy_OC(T, 1E5, phasenames[i], False, enforceGridMinimizerForLocalEq) for i in range(len(phasenames))
    ]
    alphafuncs, betafuncs = [each.chemicalpotential for each in model_phase]

    sigma_model = SigmaCoherentInterface(alphafuncs, betafuncs, mueq, vmis)

    components = [each for each in comps if each != "VA"]
    cum = int(len(components) - 1)
    print(
        "\n******************************************************************************\nOpenIEC is looking for interfacial equilibirium coposition.\nFor more information visit https://github.com/openiec/openiec."
    )
    x_s = SearchEquilibrium(sigma_model.objective, [limit] * cum, [dx] * cum)
    x_c = ComputeEquilibrium(sigma_model.objective, x_s["x"])
    print(
        "******************************************************************************\n\n"
    )
    sigma = sigma_model.infenergy(x_c)

    xx0 = [1.0 - sum(list(x0))] + list(x0)
    xx_c = [1.0 - sum(list(x_c))] + list(x_c)
    sigmapartial = list(np.array(sigma).flatten())
    sigmaavg = np.average([each for each in sigma])

    res = Dataset(
        {
            "Components": components,
            "Temperature": T,
            "Initial_Alloy_Composition": ("Components", xx0),
            "Interfacial_Composition": ("Components", xx_c),
            "Partial_Interfacial_Energy": ("Components", sigmapartial),
            "Interfacial_Energy": sigmaavg,
        }
    )

    return res

