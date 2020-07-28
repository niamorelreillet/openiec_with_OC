import unittest
import numpy as np
import os
from pycalphad import Database
from openiec.property.coherentenergy import CoherentGibbsEnergy
from openiec.property.coherentenergy_OC import CoherentGibbsEnergy_OC
from openiec.calculate.calcsigma_OC import SigmaCoherent_OC

@unittest.skipUnless(os.path.exists(os.environ.get('TDBDATA','')+'/NiAlCrHuang1999.tdb'), 'requires NiAlCrHuang1999.tdb database')
class test_coherent_NiAlCr(unittest.TestCase):
	
	__verbosity = False;
	
	def setUp(self):
		print("### test Ni-Al-Cr coherent interface between FCC_A1 and GAMMA_PRIME phases ###\n")
		# tdb filepath
		self.__tdbFile=os.environ['TDBDATA']+'/NiAlCrHuang1999.tdb'
		# components
		self.__comps = ['NI', 'AL', 'CR', 'VA']
		CoherentGibbsEnergy_OC.initOC(self.__tdbFile, self.__comps)
		
	def test_TwoPhaseEquilibrium(self):
		print("### test_TwoPhaseEquilibrium ###\n")
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 1273.0
		P = 1E5
		# instantiate two-phase OC "model"
		model_OC = CoherentGibbsEnergy_OC(T, P, phasenames, self.__verbosity)
		# instantiate two-phase pyCalphad "model"
		model = CoherentGibbsEnergy(T, Database(self.__tdbFile), self.__comps, phasenames)
		# calculate and compare chemical potentials
		x = [0.18, 0.0081]   # Al, Cr content (component molar fractions excluding the first one in comps)
		G = model.Gibbsenergy(x)
		mu = model.chemicalpotential(x)
		print('pyCalphad: ', G, mu)
		mu_OC = model_OC.chemicalpotential(x)
		G_OC = model_OC.getGibbsEnergy()
		print('OpenCalphad:', G_OC, mu_OC)
		np.testing.assert_array_almost_equal(G,G_OC,decimal=2)
		np.testing.assert_array_almost_equal(mu,mu_OC,decimal=1)
		
	def test_SinglePhaseEquilibrium(self):
		print("### test_SinglePhaseEquilibrium ###\n")
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 1273.0
		P = 1E5
		for phasename in phasenames:
			# instantiate two-phase OC "model"
			if (phasename=='GAMMA_PRIME'):
				model_OC = CoherentGibbsEnergy_OC(T, P, phasename, self.__verbosity, True) # enforce gridminizer even in this 'local' equilibrium in order to reproduce same results as pyCalphad
			else:
				model_OC = CoherentGibbsEnergy_OC(T, P, phasename, self.__verbosity,  False)
			# instantiate two-phase pyCalphad "model"
			model = CoherentGibbsEnergy(T, Database(self.__tdbFile), self.__comps, phasename)
			# calculate and compare chemical potentials
			x = [0.1931, 0.008863]   # Al, Cr content (component molar fractions excluding the first one in comps)
			G = model.Gibbsenergy(x)
			mu = model.chemicalpotential(x)
			print('pyCalphad: ', G, mu)
			mu_OC = model_OC.chemicalpotential(x)
			G_OC = model_OC.getGibbsEnergy()
			print('OpenCalphad:', G_OC, mu_OC)
			np.testing.assert_array_almost_equal(G,G_OC,decimal=2)
			np.testing.assert_array_almost_equal(mu,mu_OC,decimal=1)
				
	def test_WithSigmaCoherent(self):
		print("### test_WithSigmaCoherent ###\n")
    	# phase names
		phasenames = ['FCC_A1', 'GAMMA_PRIME']
		# temperature and pressure
		T = 1273.0
		P = 1E5
		# Given initial alloy composition. x0 corresponds to the mole fractions of Al and Cr.
		x0 = [0.180000, 0.008100]  	
		# Molar volumes of pure components to construct corresponding molar volume database.
		# Molar volume of Al.
		val = "10.269*10.0**(-6.0) + (3.860*10.0**(-5)*10.0**(-6.0))*T**1.491"
		# Molar volume of Ni.
		vni = "6.718*10.0**(-6.0) + (2.936*10.0**(-5)*10.0**(-6.0))*T**1.355"
		# Molar volume of Cr.
		vcr = "7.23*10.0**(-6.0)"
		purevms = [[vni, val, vcr], ]*2
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
			enforceGridMinimizerForLocalEq=False
		)
    
		# Print the calculated interfacial energy with xarray.Dataset type.
		print(sigma, "\n")
		# Print the calculated interfacial energy with xarray.DataArray type.
		print(sigma.Interfacial_Energy, "\n")
		# Print the calculated interfacial energy value.
		print(sigma.Interfacial_Energy.values, "\n")
		np.testing.assert_array_almost_equal(0.023783372796539266,sigma.Interfacial_Energy.values,decimal=5)
		np.testing.assert_array_almost_equal([0.798, 0.1931, 0.008863],sigma.Interfacial_Composition.values,decimal=4)
		
if __name__ == '__main__':

    unittest.main()
