# Code modified from https://github.com/open-compass/VLMEvalKit/tree/main/vlmeval/api
try:
    import torch
except ImportError:
    pass

from .smp import *
from .api import *
from .config import *

load_env()

__version__ = '0.2rc1'
