import numpy as np
import xarray as xr

class MACHINELEARNING():
    """
        Set of rules to be followed for the ML module of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
 
    def __init__(self, dataset, config):

        """ Initializing Method """

        self.dataset = dataset
        self.application = config['essentials']['application']
        self.model = config['essentials']['model']
        self.keys = config['essentials']['output_keys']
        self.output_dims = ['tag', config['essentials']['output_name']]
        self.results = "No model has been trained!"

        
    def __str__(self):

        """ Overwrite print function """

        print('-----------------------------------------------------')
        print('               MACHINE LEARNING INFO                 ')
        print('-----------------------------------------------------')
        print('\n')
        print('Machine Learning Application :', self.application)
        print('Machine Learning Model       :', self.model)
        print('Machine Learning Dataset     :', self.dataset)
        print('Machine Learning Results     :', self.results)
        return '\n'

    def adjust_dataset():

        """ Method: A must call for dataset manipulation for an ML package """


    def put_results(self, data):

        """ Method: A must call to fill the empty experiment sheet """

        self.results= xr.Dataset({'ML-Output':\
                xr.DataArray(data ,coords=[np.arange(0,len(data)),self.keys],\
                dims=self.output_dims)})

    def train(self):

        """ Method: Define set of things to be done """

        pass

    def save_model(self,name):
        
        """ Method: To save the model  """
        
        pass

    def save_result(self,name):
        
        """ Method: To save the result"""
        
        self.results.to_netcdf(name)
