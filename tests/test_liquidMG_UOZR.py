import unittest
import numpy as np
import os
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC

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

if __name__ == '__main__':

    unittest.main()
