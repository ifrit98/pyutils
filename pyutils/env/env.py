##############################################################################
# Environment Management                                                     #
##############################################################################

import os
import yaml
import pickle

getwd = os.getcwd

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))


def setwd(path):
    owd = os.getcwd()
    os.chdir(path)
    return owd

def add_to_namespace(x, **kwargs):
    if not hasattr(x, '__dict__'):
        raise ValueError(
            "Cannot update nonexistant `__dict__` for object of type {}".format(type(x)))
    x.__dict__.update(kwargs)
    return x

def add_to_namespace_dict(x, _dict):
    x.__dict__.update(_dict)
    return x

def exists_here(object_str):
    if str(object_str) != object_str:
        print("Warning: Object passed in was not a string, and may have unexpected behvaior")
    return object_str in list(globals())

def stopifnot(predicate, **kwargs):
    locals().update(kwargs) # <-- inject necessary variables into local scope to check?
    predicate_str = predicate
    if is_strlike(predicate):
        predicate = eval(predicate)
    if is_bool(predicate) and predicate not in [True, 1]:
        import sys
        sys.exit("\nPredicate:\n\n  {}\n\n is not True... exiting.".format(
            predicate_str))

def add_to_globals(x):
    if type(x) == dict:
        if not all(list(map(lambda k: is_strlike(k), list(x)))):
            raise KeyError("dict `x` must only contain keys of type `str`")
    elif type(x) == list:
        if type(x[0]) == tuple:
            if not all(list(map(lambda t: is_strlike(t[0]), x))):
                raise ValueError("1st element of each tuple must be of type 'str'")
            x = dict(x)
        else:
            raise ValueError("`x` must be either a `list` of `tuple` pairs, or `dict`")
    globals().update(x)


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __iter__(self):
        return iter(self.__dict__.items())
    def add_to_namespace(self, **kwargs):
        self.__dict__.update(kwargs)

def env(**kwargs):
    return Namespace(**kwargs)
environment = env

def parse_gitignore():
    with open('.gitignore', 'r') as f:
        x = f.readlines()
    ignore = list(map(lambda s: s.split('\n')[:-1], x))
    ignore[-1] = [x[-1]]
    return ', '.join(unlist(ignore))


is_function = lambda x: x.__class__.__name__ == 'function'
if_func = is_function

timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))

def which_os():
    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macOS"
    elif platform == "win32":
        return "windows"
    else:
        raise ValueError("Mystery os...")

def on_windows():
    return which_os() == "windows"

def on_linux():
    return which_os() == "linux"

def on_mac():
    return which_os() == "macOS"

def import_flags(path=None):
    if path is not None:
        try:
            with open(path, 'r') as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
        except:
            pass
    possible_dirs = ['./', './config', './data', './fr_train']
    for directory in possible_dirs:
        path = os.path.join(directory, 'flags.yaml')    
        FLAGS_FILE = os.path.abspath(path)
        if os.path.exists(FLAGS_FILE):
            with open(FLAGS_FILE, 'r') as f:
                return yaml.load(f, Loader=yaml.SafeLoader)
    raise ValueError("No flags file found.")

def import_history(path='history/model_history'):
    import pickle
    with open(path, 'rb') as f:
      history = pickle.load(f)
    return history


