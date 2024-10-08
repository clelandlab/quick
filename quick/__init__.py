import os

def get_version():
    with open(os.path.join(os.path.dirname(__file__), 'VERSION')) as version_file:
        return version_file.read().strip()

__version__ = get_version()

from .helper import *
from .mercator import Mercator
import quick.experiment as experiment
import quick.auto as auto
