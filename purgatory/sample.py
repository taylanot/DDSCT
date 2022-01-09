from importlib import import_module
import sys 

def sample(config, path=None):
    if config['application'] == 'SALib':
        from .doe import SALib
    else:
        raise Exception('Your sampling application is not registered!')
    
    try:
        klass = getattr(eval(config['application']), config['application'])
    except:
        raise Exception('Make sure your class name and module name are same!')
    instance = klass(config)
    sampled = instance.sample()
    print(instance)
    if path == None:
        instance.save('dataset.doe')
    else:
        instance.save(path+'/dataset.doe')
    return sampled

    
    
