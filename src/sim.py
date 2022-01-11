import numpy as np
import xarray as xr

class SIMULATION():
    """
        Set of rules to be followed for the FEM module of your choice

        * Inherit from this class to your module in order to use its 
            functionalities.
    """
 
    def __init__(self, doe, config):

        """ Initializing Method """

        self.doe = doe                                  
        self.application = config['essentials']['application']
        self.model = config['essentials']['model']
        self.num = len(doe['doe'])
        self.keys = config['essentials']['output_keys']
        self.dataset = 'No experiment has been run!'
        
    def __str__(self):

        """ Overwrite print function """

        print('-----------------------------------------------------')
        print('                  SIMULATION INFO                    ')
        print('-----------------------------------------------------')
        print('\n')
        print('Simulation Application   :', self.application)
        print('Simulation Model         :', self.model)
        print('Simulation DOE           :', self.doe)
        print('Simulation Dataset       :', self.dataset)
        return '\n'

    def _call_counter(func):
        def helper(*args, **kwargs):
            helper.calls += 1
            return func(*args, **kwargs)
        helper.calls = 0
        return helper

    @_call_counter
    def __call__(self, _id):

        """ Method: Run the Experiment with given ID and add it to database """

        return self.run_experiment(_id)


    def put_results(self, results):

        """ Method: A must call to fill the empty experiment sheet """

        self.doe['Results'] = xr.DataArray(results,\
                coords=[np.arange(0,self.num),self.keys],\
                dims=['doe','outputs'])

        self.dataset = self.doe

    def run_experiment(self, _id):

        """ Method: Define set of things to be done """

        pass

    def save(self,name):
        
        """ Method: To save the results to a file """
        
        self.dataset.to_netcdf(name)
