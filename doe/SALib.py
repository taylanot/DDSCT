from SALib.sample import sobol_sequence
from ..src.doe import DOE

class SALibSampler(DOE):
    """
        DOE-Module wrap for SALib 
    """
 
    def __init__(self, config):

        """ Initialize """

        super().__init__(config)                  # Initialize base-class

        self.method = config['additional']['method']

    def sample(self):

        """ Method: Overwrite sampling method """

        func = getattr(self,'_'+self.method) # Select your method of sampling
        self.data = func()
        

    def _sobol(self):

        """ Method: Sobol sequence generator """

        points = sobol_sequence.sample(self.num,self.dim)                   # Create [0,1] self.dim-dimnesional hypercube
        for i, bound in enumerate(self.bounds):                  # Stretch the hypercube towards your bounds
            points[:,i] = points[:,i] * (bound[1] - bound[0]) + bound[0]    
        return points

    
        



        

    
