#from F3DASM.src.doe import *
#from F3DASM.doe.SALib import SALib
#from F3DASM.sample import *
#config = dict()
##seed = 24 
#config['doe'] = {
#        'descriptors':{'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]},
#        'application': 'SALib',
#        'method': 'sobol',
#        'experiment_number': 100}
#
#dataset = sample(config['doe'])

from sacred import Experiment
ex = Experiment('my_experiment')

@ex.config
def my_config():
    config = dict()
    config['doe'] = {
            'descriptors':{'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]},
            'application': 'SALib',
            'method': 'sobol',
            'experiment_number': 100}
    
def some_function(conf):
    print(conf)

@ex.main
def my_main(config):
    print(config)
    some_function(config[
