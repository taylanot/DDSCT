from DDSCT.src.doe import *
from DDSCT.src.sim import *
from DDSCT.doe.SALib import SALibSampler
from DDSCT.main import *
import numpy as np
import torch

from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('Bessa2017')

class Simpleton(torch.nn.Module):
    def __init__(self,set_up):
        super(Simpleton, self).__init__()

        for k, v in set_up.items():
            setattr(self, k, v)

        self.hidden = torch.nn.Linear(self.n_feature, self.n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(self.n_hidden, self.n_output)   # output layer
        self.activation = torch.nn.ReLU( )

    def forward(self, x):
        x = self.activation(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x
@ex.config
def my_config():
    config = dict()
    config['doe'] = {'essentials':
            {
            'descriptors':{'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1],\
                    'L':[1,2],'r':[0.1, 0.3],'Vf':[0.1, 0.2]},
            'application': 'SALib',
            'model': 'SALibSampler',
            'experiment_number': 10
            },
            'additional':
            {
            'method': 'sobol'
            }}


    config['simulation'] = {'essentials':
                {
                'application': 'SolidMechanicsTools',
                'model': 'Bessa2017',
                'output_keys': ['S11', 'S12', 'S22']
                }}

    config['machinelearning'] = {'essentials':
                {
                'application': 'PyTorch',
                'model': 'SimpleTraining',
                'output_name': 'MSE-Loss',
                'output_keys': ['train_loss','test_loss']
                },
                'additional':
                {
                'network': Simpleton,
                'network_parameters':{'n_feature':6, 'n_hidden':10, 'n_output':3},
                'split': 0.2,
                'optimizer': torch.optim.SGD,
                'optimizer_parameters':{'lr':0.01},
                'loss': torch.nn.MSELoss,
                'epoch': 100,
                }}
    NAME = "Bessa2017_DDSCT"
    ex.observers.append(FileStorageObserver(NAME))

@ex.capture
def get_info(_run):
    return _run._id, _run.experiment_info["name"]

@ex.automain
def main(config,NAME):
    _id, _ = get_info()
    artifacts = os.path.join(NAME,'artifacts',_id)
    os.makedirs(artifacts)
    doe, _ = sample(config['doe'],path=os.path.join(artifacts))
    dataset, _ = simulate(doe, config['simulation'], num_workers=8, path=os.path.join(artifacts))
    surrogate = train(dataset, config['machinelearning'], path=os.path.join(artifacts))

#sim = Test(doe, config['simulation'])
#print(sim)
#pool = Pool(processes=8)
#
#for output in tqdm.tqdm(pool.imap_unordered(sim, doe['doe'].values), total=len(doe['doe'].values)):
#    print(output)

#from sacred import Experiment
#ex = Experiment('my_experiment')

#@ex.config
#def my_config():
#    config = dict()
#    config['doe'] = {
#            'descriptors':{'F11':[-0.15, 1], 'F12':[-0.1,0.15],'F22':[-0.15, 1]},
#            'application': 'SALib',
#            'method': 'sobol',
#            'experiment_number': 100}
#    
#def some_function(conf):
#    print(conf)
#
#@ex.automain
#def my_main(_run,config):
#    print(config)
#    some_function(config['doe'])
#    print(_run.experiment_info['name'])
