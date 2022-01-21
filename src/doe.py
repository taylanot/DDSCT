import numpy as np
import xarray as xr 

class DOE():
    """
        Set of rules to be followed for the DOE module of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
    def __init__(self,config):

        """ Initialize """

        self.config= config                             
        self.descriptors = config['essentials']['descriptors']
        self.application = config['essentials']['application']        
        self.model = config['essentials']['model']        
        self.dim = len(self.descriptors)                   
        self.num = config['essentials']['experiment_number']
        self.keys = list(self.descriptors.keys())          
        self.bounds = list(self.descriptors.values())          
        self.dataset = 'Not sampled yet!'

    def __str__(self):

        """ Overwrite print function """

        print('-----------------------------------------------------')
        print('                       DOE INFO                      ')
        print('-----------------------------------------------------')
        print('\n')
        print('Sampling Application :',self.application)
        print('Sampling Model       :',self.model)
        print('Number of Descriptors:',self.dim)
        print('Number of Experiments:',self.num)
        print('DOE Dataset          :',self.dataset)
        return '\n'

    def save(self,name):

        """ Method: Pickle the doe points """  

        self.dataset.to_netcdf(name)

    def sample(self):

        """ Method: Method for Sampling """  
        pass 

    def put_results(self):

        """ Method: DOE Dataset Creation """  

    
        self.dataset = xr.Dataset({'Running_Variables':\
                xr.DataArray(self.data,coords=[np.arange(0,self.num),self.keys],\
                dims=['doe','descriptor'])})
        return self.dataset





