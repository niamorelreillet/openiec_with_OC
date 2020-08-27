import pyOC
import numpy as np
from pyOC import opencalphad as oc
from pyOC import PhaseStatus as phStat
from pyOC import GridMinimizerStatus as gmStat

class CoherentGibbsEnergy_OC(object):
    __db = None
    __comps = None
    __verbosity = None
    def __identity(x):
        return x
    
    @classmethod
    def initOC(cls, db, comps):
        cls.__db  = db
        cls.__comps = comps
        # read database
        oc.readtdb(cls.__db, tuple([comp for comp in cls.__comps if comp != 'VA']))

    def __init__(self, T, P, phasenames, verbosity=False, enforceGridMinimizerForLocalEq=False):
        if (self.__db==None):
            raise RuntimeError('database has not been set, class method initOC should be called first')
        self.__T = T
        self.__P = P
        if (type(phasenames) is list):
            self.__phasenames = phasenames
        else:
            self.__phasenames = [phasenames]
        # see in which case we are: global or local equilibrium
        if (len(self.__phasenames)==2):
            self.__gridMinimizerStatus = gmStat.On
        elif(len(self.__phasenames)==1):
            if (enforceGridMinimizerForLocalEq):
                self.__gridMinimizerStatus = gmStat.On
            else:
                self.__gridMinimizerStatus = gmStat.Off
        else:
            raise ValueError('invalid number of phases provided (should be one or two)')
        # set verbosity
        oc.setVerbosity(verbosity)
        self.__eq_val = None
            
    def __eqfunc(self, x):
        # suspend all other phases
        oc.setPhasesStatus(('* ',), phStat.Suspended)
        oc.setPhasesStatus(tuple(self.__phasenames), phStat.Entered, 1.0)
        # set temperature and pressure
        oc.setTemperature(self.__T)
        oc.setPressure(self.__P)
        # set initial molar amounts
        oc.setElementMolarAmounts(self.__calculateMolarAmounts(x))
        # calculate equilibrium
        oc.calculateEquilibrium(self.__gridMinimizerStatus)

    def __calculateMolarAmounts(self, x):
        elementMolarAmounts = {}
        xSum = 0.0
        for i in range(len(x)):
            xSum += x[i]
        elementMolarAmounts[self.__comps[0]]=1.0-xSum
        count = 1 
        for i in range(len(x)):
            if (self.__comps[i]=='VA'):
                count=count+1
            elementMolarAmounts[self.__comps[count]]=x[i]
            count=count+1
        return elementMolarAmounts
        
    def getGibbsEnergy(self):
        self.__eq_val = [ oc.getGibbsEnergy() ]
        return self.__eq_val

    def getChemicalPotentials(self):
        mu = oc.getChemicalPotentials()
        self.__eq_val = [ mu[comp] for comp in self.__comps if comp != 'VA' ]
        return self.__eq_val
    
    # molar volume related methods  
    def constantPartialMolarVolumeFunctions(self, x, constituentMassDensityLaws, epsilon=1E-6):
        partialMolarVolumes,exactVolume,approxVolume = self.calculatePartialMolarVolume(self.__calculateMolarAmounts(x), constituentMassDensityLaws, epsilon)
        volumeError=(approxVolume/exactVolume-1.0)*100.0
        if (abs(volumeError)>1E-4):
            print(volumeError,approxVolume,exactVolume)
            #raise RuntimeError('volume error is too high')
        return [ (lambda comp : lambda _: partialMolarVolumes[comp])(comp) for comp in self.__comps if comp != 'VA' ]
                     
    ## evaluate partial molar volumes by an approximation of the first order volume derivative by a second-order finite difference formula
    def calculatePartialMolarVolume(self,elementMolarAmounts,constituentMassDensityLaws,epsilon):
        print(elementMolarAmounts, epsilon)
        # suspend all other phases
        oc.setPhasesStatus(('* ',), phStat.Suspended)
        oc.setPhasesStatus(tuple(self.__phasenames), phStat.Entered, 1.0)
        # set pressure
        oc.setPressure(self.__P)
        # set temperature
        oc.setTemperature(self.__T)
        # evaluate volume
        oc.setElementMolarAmounts(elementMolarAmounts)
        oc.calculateEquilibrium(gmStat.Off)
        exactVolume = self.__calculateVolume(oc.getPhasesAtEquilibrium().getPhaseConstituentComposition(),oc.getConstituentsDescription(),constituentMassDensityLaws)
        # evaluate (elementwise) partial molar volume (approximation of first order volume derivative by a second-order finite difference formula)
        partialMolarVolumes={}
        for element in elementMolarAmounts:
            # evaluate volume for n[element]+epsilone
            modifiedElementMolarAmounts=elementMolarAmounts.copy()
            modifiedElementMolarAmounts[element] += epsilon
            print(element, modifiedElementMolarAmounts)
            oc.setElementMolarAmounts(modifiedElementMolarAmounts)
            oc.calculateEquilibrium(gmStat.Off)
            volumePlus = self.__calculateVolume(oc.getPhasesAtEquilibrium().getPhaseConstituentComposition(),oc.getConstituentsDescription(),constituentMassDensityLaws)
            # evaluate volume for n[element]-epsilone
            modifiedElementMolarAmounts[element] -= 2.0*epsilon
            print(element, modifiedElementMolarAmounts)
            oc.setElementMolarAmounts(modifiedElementMolarAmounts)
            oc.calculateEquilibrium(gmStat.Off)
            volumeMinus = self.__calculateVolume(oc.getPhasesAtEquilibrium().getPhaseConstituentComposition(),oc.getConstituentsDescription(),constituentMassDensityLaws)
            partialMolarVolumes[element]=(volumePlus-volumeMinus)/(2.0*epsilon)
        # calculate approximate volume from partial volumes
        approxVolume = 0.0
        for element, molarAmount in oc.getPhasesAtEquilibrium().getPhaseElementComposition()[list(oc.getPhasesAtEquilibrium().getPhaseElementComposition())[0]].items():
            approxVolume+=molarAmount*partialMolarVolumes[element]
        return partialMolarVolumes,exactVolume,approxVolume
        
    ## convert constituent molar fractions to mass fractions
    def __convertConstituentMolarToMassFractions(self,constituentMolarFractions,constituentsDescription):
        constituentMassFractions=constituentMolarFractions.copy()
        tot=0.0
        for constituent in constituentMassFractions:
            constituentMassFractions[constituent] *= constituentsDescription[constituent]['mass']
            tot += constituentMassFractions[constituent]
        fac = 1.0/tot
        for constituent in constituentMassFractions:
            constituentMassFractions[constituent] *= fac
        return constituentMassFractions     

    ## function to calculate molar volume from constituent 
    def __calculateVolume(self,phaseConstituentComposition,constituentsDescription,constituentMassDensityLaws):
        print(phaseConstituentComposition.keys())
        if (len(phaseConstituentComposition) != 1):
            print('error: not a single phase (%s) at equilibrium for molar volume calculation' % list(phaseConstituentComposition.keys()))
            exit()
        constituentMolarFractions=phaseConstituentComposition[list(phaseConstituentComposition.keys())[0]]
        # mass fractions from molar fractions
        constituentMassFractions=self.__convertConstituentMolarToMassFractions(constituentMolarFractions,constituentsDescription)
        # ideal mixing law to evaluate mixture density
        density=0.0
        for constituent, massFraction in constituentMassFractions.items():
            density += massFraction/constituentMassDensityLaws[constituent](self.__T)
        density=1.0/density
        # total mass (mass is 'B' in OC)
        mass=oc.getScalarResult('B')
        return mass*1E-3/density
        
# methods that are common with CoherentGibbsEnergy class (based on pyCalphad)
    def Gibbsenergy(self, x):
        self.__eqfunc(x)
        return self.getGibbsEnergy()
        
    def chemicalpotential(self, x):
        self.__eqfunc(x)
        return self.getChemicalPotentials();

