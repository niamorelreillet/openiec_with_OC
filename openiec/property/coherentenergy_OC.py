import pyOC
import numpy as np
from pyOC import opencalphad as oc
from pyOC import PhaseStatus as phStat
from pyOC import GridMinimizerStatus as gmStat

class CoherentGibbsEnergy_OC(object):
    __db = None
    __comps = None
    __verbosity = None
    
    @classmethod
    def initOC(cls, db, comps, verbosity):
        cls.__db  = db
        cls.__comps = comps
        cls.__verbosity = verbosity
        # set verbosity
        oc.setVerbosity(cls.__verbosity)
        # read database
        oc.readtdb(cls.__db, tuple([comp for comp in cls.__comps if comp != 'VA']))

    def __init__(self, T, P, phasenames, enforceGridMinimizerForLocalEq=False):
        if (self.__db==None):
            raise RuntimeError('database has not been set, class method initOC should be called first')
        self.__T = T
        self.__P = P
        if (type(phasenames) is list):
            self.__phasenames = phasenames
        else:
            self.__phasenames = [phasenames]
        # see in which case we are: global or local equilibrium
        print(self.__phasenames, len(self.__phasenames))
        if (len(self.__phasenames)==2):
            self.__gridMinimizerStatus = gmStat.On
        elif(len(self.__phasenames)==1):
            if (enforceGridMinimizerForLocalEq):
                self.__gridMinimizerStatus = gmStat.On
            else:
                self.__gridMinimizerStatus = gmStat.Off
        else:
            raise ValueError('invalid number of phases provided (should be one or two)')
        self.__eq_val = None
            
    def __eqfunc(self, x):
        # suspend all other phases
        oc.setPhasesStatus(('* ',), phStat.Suspended)
        oc.setPhasesStatus(tuple(self.__phasenames), phStat.Entered, 1.0)
        # set temperature and pressure
        oc.setTemperature(self.__T)
        oc.setPressure(self.__P)
        # set initial molar amounts
        elementMolarAmounts = {}
        xSum = 0.0
        count = 1 
        for i in range(len(x)):
            xSum += x[i]
            if (self.__comps[i]=='VA'):
                count=count+1
            elementMolarAmounts[self.__comps[count]]=x[i]
            count=count+1
        elementMolarAmounts[self.__comps[0]]=1.0-xSum
        print(elementMolarAmounts)
        oc.setElementMolarAmounts(elementMolarAmounts)
        # calculate equilibrium
        oc.calculateEquilibrium(self.__gridMinimizerStatus)

    def getGibbsEnergy(self):
        self.__eq_val = [ oc.getGibbsEnergy() ]
        return self.__eq_val

    def getChemicalPotentials(self):
        mu = oc.getChemicalPotentials()
        self.__eq_val = [ mu[comp] for comp in self.__comps if comp != 'VA' ]
        return self.__eq_val
        
# methods that are common with CoherentGibbsEnergy class (based on pyCalphad)
    def Gibbsenergy(self, x):
        self.__eqfunc(x)
        return self.getGibbsEnergy()
        
    def chemicalpotential(self, x):
        self.__eqfunc(x)
        return self.getChemicalPotentials();

