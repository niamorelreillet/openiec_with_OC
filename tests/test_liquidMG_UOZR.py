import unittest
import numpy as np
import os
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC
from openiec.calculate.calcsigma_OC import SigmaCoherent_OC

@unittest.skipUnless(os.path.exists(os.environ.get('TDBDATA_PRIVATE','')+'/feouzr.tdb'), 'requires feouzr.tdb database')
class test_liqMG_UOZr(unittest.TestCase):
    
    __verbosity = False;
    
    def setUp(self):
        print("### test U-O-Zr coherent interface in the liquid miscibility gap ###\n")
        # tdb filepath
        self.__tdbFile=os.environ['TDBDATA_PRIVATE']+'/feouzr.tdb'
        # components
        self.__comps = ['O', 'U', 'ZR']
        # mass density laws (from Barrachin2004)
        self.__constituentDensityLaws = {
            'U1'   : lambda T: 17270.0-1.358*(T-1408),
            'ZR1'  : lambda T: 6844.51-0.609898*T+2.05008E-4*T**2-4.47829E-8*T**3+3.26469E-12*T**4,
            'O2U1' : lambda T: 8860.0-9.285E-1*(T-3120),
            'O2ZR1': lambda T: 5150-0.445*(T-2983),
            'O1'   : lambda T: 1.141 # set to meaningless value but ok as, no 'free' oxygen in the considered mixtures
        }
        CoherentGibbsEnergy_OC.initOC(self.__tdbFile, self.__comps)

    def test_MolarVolume(self):
        print("### test_MolarVolume ###\n")
        # phase names
        phasenames = ['LIQUID']
        # temperature and pressure
        T = 2000.00
        P = 1E5
        # instantiate OC "model"
        model = CoherentGibbsEnergy_OC(T, P, phasenames, self.__verbosity)
        # calculate and compare chemical potentials
        x = [0.343298, 0.241778]   # U, Zr content (component molar fractions excluding the first one in comps)
        #
        functions=model.constantPartialMolarVolumeFunctions(x, self.__constituentDensityLaws)
        np.testing.assert_allclose(0.005934892340596354, functions[0](x), rtol=1e-6, atol=1E-6)
        np.testing.assert_allclose(0.015389805158548542, functions[1](x), rtol=1e-6, atol=1E-6)
        np.testing.assert_allclose(0.012607127224985304, functions[2](x), rtol=1e-6, atol=1E-6)
        
    # not ready yet!
    def test_WithSigmaCoherent(self):
        print("### test_WithSigmaCoherent ###\n")
        # phase names
        phasenames = ['LIQUID', 'LIQUID']
        # temperature and pressure
        T = 3000.0
        P = 1E5
        # Given initial alloy composition. x0 is the mole fraction of U and Zr.
        x0 = [0.343298, 0.241778]      
        # Molar volumes of pure components evaluated at x0 and kept constant afterwards
        model = CoherentGibbsEnergy_OC(T, P, phasenames[0], self.__verbosity)
        functions=model.constantPartialMolarVolumeFunctions(x0, self.__constituentDensityLaws, 1E-1)
        functions=model.constantPartialMolarVolumeFunctions(x0, self.__constituentDensityLaws, 1E-2)
        functions=model.constantPartialMolarVolumeFunctions(x0, self.__constituentDensityLaws, 1E-3)
        functions=model.constantPartialMolarVolumeFunctions(x0, self.__constituentDensityLaws, 1E-4)
        functions=model.constantPartialMolarVolumeFunctions(x0, self.__constituentDensityLaws, 1E-5)
        # Composition range for searching initial interfacial equilibrium composition.
        limit = [0.0001, 0.9]
        # Composition step for searching initial interfacial equilibrium composition.
        dx = 0.1
    
        # calculate interfacial energy
        sigma = SigmaCoherent_OC(
            T=T,
            x0=x0,
            db=self.__tdbFile,
            comps=self.__comps,
            phasenames=phasenames,
            purevms=functions,
            limit=limit,
            dx=dx,
            enforceGridMinimizerForLocalEq=True
        )
    
        # Print the calculated interfacial energy with xarray.Dataset type.
        print(sigma, "\n")
        # Print the calculated interfacial energy with xarray.DataArray type.
        print(sigma.Interfacial_Energy, "\n")
        # Print the calculated interfacial energy value.
        print(sigma.Interfacial_Energy.values, "\n")
        
if __name__ == '__main__':

    unittest.main()
