import numpy as np
from ..src.sim import SIMULATION

class test(SIMULATION):
    def __init__(self, doe, config):
       super().__init__(doe, config)
       self.__name__ = 'test'
    def run_experiment(self, _id):
        print(self.doe['Running_Variables'][_id].values*_id)
        return  self.doe['Running_Variables'][_id].values*_id


    

