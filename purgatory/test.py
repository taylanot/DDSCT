from sacred import Experiment
from .f3dasm.FENiCS.Bessa2017 import simulate
from .f3dasm.ml.PyTorch import train 
from .f3dasm.doe.SALib import sample
from .SolidMechanics import Bessa2017


ex = Experiment('test-f3dasm')

@ex.config
def config():
    config = dict()
    #seed = 24 
    config['doe'] = {
            'descriptors':{'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]},
            'application': 'SALib',
            'method': 'sobol',
            'experiment_number': '10'}

    config['simulation'] = {
            'application': 'FENiCS/SolidMechanics',
            'model' = Bessa2017}

    config['machinelearning'] = {
            'application': 'PyTorch',
            'model':SimpleNet}


@ex.automain
def main(config):
    doe = sample(config['doe'])
    # -> save doe 
    dataset = simulate(doe , config['simulation'])
    # -> save dataset
    # -> save dataset
    #surrogate = train(dataset, config['machinelearning'])
    # -> save surrogate

