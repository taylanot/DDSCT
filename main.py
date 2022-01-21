from importlib import import_module
import sys 
import os 
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm

def sample(config, path=None):
    if config['essentials']['application'] == 'SALib':
        from .doe import SALib
    else:
        raise Exception('Your sampling application is not registered!')
    
    try:
        klass = eval(config['essentials']['application']+'.'+config['essentials']['model'])
    except:
        raise Exception('Make sure your your model is in the application file!')

    instance = klass(config)
    instance.sample()
    sampled = instance.put_results()
    print(instance)

    if path == None:
        instance.save('dataset.doe')
    else:
        instance.save(path+'/dataset.doe')
    return sampled, instance

def simulate(doe, config, num_workers=None, path=None):

    if num_workers == None or num_workers >= os.cpu_count():
        num_workers = os.cpu_count()

    if config['essentials']['application'] == 'test':
        from .sim import test
    if config['essentials']['application'] == 'SolidMechanicsTools':
        from .sim import SolidMechanicsTools
    else:
        raise Exception('Your sampling application is not registered!')

    try:
        klass = eval(config['essentials']['application']+'.'+config['essentials']['model'])
    except:
        raise Exception('Make sure your your model is in the application file!')

    instance = klass(doe, config, path)
    pool = Pool(processes=num_workers)
    
    print('-----------------------------------------------------')
    print('                  SIMULATION PROGRESS                ')
    print('-----------------------------------------------------')

    results = np.array(list(tqdm(pool.imap(instance, doe['doe'].values), total=len(doe['doe'].values))))
    pool.close()
    instance.put_results(results)
    print(instance)

    if path == None:
        instance.save('dataset.sim')
    else:
        instance.save(path+'/dataset.sim')
 
    return instance.dataset, instance

def train(dataset, config, path=None):

    if config['essentials']['application'] == 'PyTorch':
        from .ml import PyTorch
    else:
        raise Exception('Your sampling application is not registered!')

    try:
        klass = eval(config['essentials']['application']+'.'+config['essentials']['model'])
    except:
        raise Exception('Make sure your your model is in the application file!')

    instance = klass(dataset, config)

    print('-----------------------------------------------------')
    print('                 TRAINING PROGRESS                   ')
    print('-----------------------------------------------------')

    instance.adjust_dataset()
    results = instance.train()
    instance.put_results(results)

    print(instance)

    if path == None:
        instance.save_result('dataset.ml')
        instance.save_model('trained'+str(config['essentials']['model'])+'.ml')
    else:
        instance.save_result(path+'/dataset.ml')
        instance.save_model(path+'/trained'+str(config['essentials']['model'])+'.ml')

 
    return instance.dataset, instance
