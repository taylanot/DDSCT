from SolidMechanicsTools.src.domain import *
from SolidMechanicsTools.models.rve import *
from SolidMechanicsTools.models.rve_utils import *

import pandas as pd
import matplotlib.pyplot as plt

import torch 
import pandas as pd
import os 

from ..src.sim import SIMULATION
import itertools, sys
import xarray as xr
spinner = itertools.cycle(['-', '/', '|', '\\'])

set_log_level(30)
class Bessa2017(SIMULATION):
    def __init__(self, doe, config, path=None):
        super().__init__(doe, config)
        self.__name__ = 'Bessa2017b'
        rves = GENERATE_RVES(doe, config, path)

        dirs = np.empty(self.num,dtype=object)
                
        sys.stdout.write('Creating the RVEs...')
        for i in (range(self.num)):
            sys.stdout.write(next(spinner))   
            sys.stdout.flush()                
            sys.stdout.write('\b')            
            dirs[i] = rves.run_experiment(i)
        sys.stdout.write('\n')

        self.doe['RVE_Directory'] = xr.DataArray(dirs,\
                coords=[np.arange(0,self.num)],\
                dims=['doe'])

    def run_experiment(self, _id):

        F11 = self.doe.sel(descriptor='F11', doe=_id)['Running_Variables'].values
        F12 = self.doe.sel(descriptor='F12', doe=_id)['Running_Variables'].values
        F22 = self.doe.sel(descriptor='F22', doe=_id)['Running_Variables'].values
        directory = self.doe.sel(doe=_id)['RVE_Directory'].values
        domain = DOMAIN(str(directory)+'.xdmf')
        FEM_model = NeoHookean_Kirchhiff_RVE(domain)

        F_macro = np.array([[F11,F12],[F12,F22]])

        S = FEM_model(F_macro,0)

        return np.array([S[0,0],S[0,1],S[1,1]])


class GENERATE_RVES(SIMULATION):
    def __init__(self, doe, config, path=None):
        super().__init__(doe, config)
        self.__name__ = 'Bessa2017b'
        self.rve = Create_RVE_gmshModel(dim=2, directory=path+'/rves')

    def run_experiment(self, _id):

        L = self.doe.sel(descriptor='L', doe=_id)['Running_Variables'].values
        r = self.doe.sel(descriptor='r', doe=_id)['Running_Variables'].values
        Vf = self.doe.sel(descriptor='Vf', doe=_id)['Running_Variables'].values

        domain, directory = self.rve(L,r,Vf)
        return directory


