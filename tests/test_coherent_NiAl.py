import unittest
import numpy as np
import os
from pycalphad import Database
from openiec.property.coherentenergy import CoherentGibbsEnergy
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC
from openiec.calculate.calcsigma_OC import SigmaCoherent_OC

@unittest.skipUnless(os.path.exists(os.environ.get('TDBDATA','')+'/NiAlHuang1999.tdb'), 'requires NiAlHuang1999.tdb database')
class test_coherent_NiAl(unittest.TestCase):
	
	__verbosity = False;
	
	def setUp(self):
		print("### test Ni-Al coherent interface between FCC_A1 and GAMMA_PRIME phases ###\n")
		# tdb filepath
		self.__tdbFile=os.environ['TDBDATA']+'/NiAlHuang1999.tdb'
		# components
		self.__comps = ['NI', 'AL', 'VA']
		CoherentGibbsEnergy_OC.initOC(self.__tdbFile, self.__comps)
		
	def test_TwoPhaseEquilibrium(self):
		print("### test_TwoPhaseEquilibrium ###\n")
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 800.00
		P = 1E5
		# instantiate two-phase OC "model"
		model_OC = CoherentGibbsEnergy_OC(T, P, phasenames, self.__verbosity)
		# instantiate two-phase pyCalphad "model"
		model = CoherentGibbsEnergy(T, Database(self.__tdbFile), self.__comps, phasenames)
		# calculate and compare chemical potentials
		x = [0.2]   # Al content (component molar fractions excluding the first one in comps)
		G = model.Gibbsenergy(x)
		mu = model.chemicalpotential(x)
		print('pyCalphad: ', G, mu)
		mu_OC = model_OC.chemicalpotential(x)
		G_OC = model_OC.getGibbsEnergy()
		print('OpenCalphad:', G_OC, mu_OC)
		np.testing.assert_array_almost_equal(G,G_OC,decimal=2)
		np.testing.assert_array_almost_equal(mu,mu_OC,decimal=2)
		
	def test_SinglePhaseEquilibrium(self):
		print("### test_SinglePhaseEquilibrium ###\n")
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 800.00
		P = 1E5
		for phasename in phasenames:
			# instantiate two-phase OC "model"
			if (phasename=='GAMMA_PRIME'):
				model_OC = CoherentGibbsEnergy_OC(T, P, phasename, self.__verbosity, True) # enforce gridminizer even in this 'local' equilibrium in order to reproduce same results as pyCalphad
			else:
				model_OC = CoherentGibbsEnergy_OC(T, P, phasename, self.__verbosity, False)
			# instantiate two-phase pyCalphad "model"
			model = CoherentGibbsEnergy(T, Database(self.__tdbFile), self.__comps, phasename)
			# calculate and compare chemical potentials
			x = [0.5]   # Al content (component molar fractions excluding the first one in comps)
			G = model.Gibbsenergy(x)
			mu = model.chemicalpotential(x)
			print('pyCalphad: ', G, mu)
			mu_OC = model_OC.chemicalpotential(x)
			G_OC = model_OC.getGibbsEnergy()
			print('OpenCalphad:', G_OC, mu_OC)
			np.testing.assert_array_almost_equal(G,G_OC,decimal=2)
			np.testing.assert_array_almost_equal(mu,mu_OC,decimal=2)
				
	def test_WithSigmaCoherent(self):
		print("### test_WithSigmaCoherent ###\n")
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 800.00
		P = 1E5
		# Given initial alloy composition. x0 is the mole fraction of Al.
		x0 = [0.2]  	
		# Molar volumes of pure components to construct corresponding molar volume database.
		# Molar volume of Ni.
		vni = "6.718*10.0**(-6.0) + (2.936*10.0**(-5)*10.0**(-6.0))*T**1.355" 
		# Molar volume of Al.
		val = "10.269*10.0**(-6.0) + (3.860*10.0**(-5)*10.0**(-6.0))*T**1.491"
		purevms = [[vni, val], ]*2
		# Composition range for searching initial interfacial equilibrium composition.
		limit = [0.0001, 0.3]
		# Composition step for searching initial interfacial equilibrium composition.
		dx = 0.1
    
		# calculate interfacial energy
		sigma = SigmaCoherent_OC(
			T=T,
			x0=x0,
			db=self.__tdbFile,
			comps=self.__comps,
			phasenames=phasenames,
			purevms=purevms,
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
		print(sigma.Interfacial_Energy.values, "\n")
		np.testing.assert_array_almost_equal(0.026624295275832557,sigma.Interfacial_Energy.values,decimal=5)
		np.testing.assert_array_almost_equal([0.88, 0.12],sigma.Interfacial_Composition.values,decimal=4)
		
if __name__ == '__main__':

    unittest.main()
