import unittest
import numpy as np
import os
from pycalphad import Database
from openiec.property.coherentenergy import CoherentGibbsEnergy
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC

@unittest.skipUnless(os.path.exists(os.environ.get('TDBDATA','')+'/NiAlHuang1999.tdb'), 'requires NiAlHuang1999.tdb database')
class testNiAl(unittest.TestCase):
	def setUp(self):
		print("### test Ni-Al coherent interface between FCC_A1 and GAMMA_PRIME phases ###\n")
		# tdb filepath
		self.__tdbFile=os.environ['TDBDATA']+'/NiAlHuang1999.tdb'
		# components
		self.__comps = ['NI', 'AL', 'VA']
		CoherentGibbsEnergy_OC.initOC(self.__tdbFile, self.__comps, False)
		
	def test_TwoPhaseEquilibrium(self):
		
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 800.00
		P = 1E5
		# instantiate two-phase OC "model"
		model_OC = CoherentGibbsEnergy_OC(T, P, phasenames)
		# instantiate two-phase pyCalphad "model"
		model = CoherentGibbsEnergy(T, Database(self.__tdbFile), self.__comps, phasenames)
		# calculate and compare chemical potentials
		x = [0.2]   # Al content (component molar fractions excluding the first one in comps)
		G = model.Gibbsenergy(x)
		mu = model.chemicalpotential(x)
		print(G, mu)
		mu_OC = model_OC.chemicalpotential(x)
		G_OC = model_OC.getGibbsEnergy()
		print(G_OC, mu_OC)
		np.testing.assert_array_almost_equal(G,G_OC,decimal=2)
		np.testing.assert_array_almost_equal(mu,mu_OC,decimal=2)
		
	def test_SinglePhaseEquilibrium(self):
		# tdb filepath
		tdbFile=os.environ['TDBDATA']+'/NiAlHuang1999.tdb'
		# components
		comps = ['NI', 'AL', 'VA']
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 800.00
		P = 1E5
		for phasename in phasenames:
			# instantiate two-phase OC "model"
			if (phasename=='GAMMA_PRIME'):
				model_OC = CoherentGibbsEnergy_OC(T, P, phasename, True) # enforce gridminizer even in this 'local' equilibrium in order to reproduce same results as pyCalphad
			else:
				model_OC = CoherentGibbsEnergy_OC(T, P, phasename, False)
			# instantiate two-phase pyCalphad "model"
			model = CoherentGibbsEnergy(T, Database(self.__tdbFile), self.__comps, phasename)
			# calculate and compare chemical potentials
			x = [0.5]   # Al content (component molar fractions excluding the first one in comps)
			G = model.Gibbsenergy(x)
			mu = model.chemicalpotential(x)
			print(G, mu)
			mu_OC = model_OC.chemicalpotential(x)
			G_OC = model_OC.getGibbsEnergy()
			print(G_OC, mu_OC)
			np.testing.assert_array_almost_equal(G,G_OC,decimal=2)
			np.testing.assert_array_almost_equal(mu,mu_OC,decimal=2)
		
if __name__ == '__main__':

    unittest.main()
